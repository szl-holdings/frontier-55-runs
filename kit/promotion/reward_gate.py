"""reward_gate.py -- post-training reward step for Salad candidate lanes.

Runs ON the training instance right after salad_worker.py finishes:
loads the base + the freshly-trained final adapter, measures teacher-forced
mean cross-entropy on an eval slice, and publishes an szl.reward/v1 receipt
to SZLHOLDINGS/frontier-rewards. Fail-closed: any error means NO receipt
(an absent reward receipt is honest absence, never a fabricated score).

This is the estate's reward loop: every new candidate model the training
fleet creates gets measured and scored, and improvement gets a promotion
recommendation receipt that pushes the frontier queue.

Honest scope: CE-delta is a teacher-forced likelihood metric, NOT human
quality. Eval slices fall back to the dataset tail when no heldout file
exists (noted in the receipt).
"""
import argparse
import datetime as dt
import json
import os
import sys

SCHEMA = "szl.reward/v1"
OUT_REPO = "SZLHOLDINGS/frontier-rewards"
MAX_ROWS = 48

EVAL_PREFS = [
    "heldout.jsonl", "eval.jsonl", "test.jsonl", "validation.jsonl",
    "data/heldout.jsonl", "data/eval.jsonl", "data/test.jsonl",
]
TRAIN_PREFS = ["train.jsonl", "data/train.jsonl"]


def pick_file(files, prefs):
    for p in prefs:
        if p in files:
            return p
    return None


def load_rows(dataset, max_rows):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(dataset, repo_type="dataset")
    jsonls = sorted(f for f in files if f.endswith(".jsonl"))
    if not jsonls:
        raise RuntimeError("dataset has no jsonl files")
    heldout = pick_file(files, EVAL_PREFS)
    source = heldout
    if source is None:
        source = pick_file(files, TRAIN_PREFS) or jsonls[0]
    path = hf_hub_download(dataset, source, repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    total = len(rows)
    if heldout is not None:
        slc = rows[:max_rows]
        note = "dedicated eval file"
    else:
        slc = rows[-max_rows:]
        note = "dataset tail slice (may overlap training data)"
    return slc, total, source, heldout is not None, note


def render(row, tok):
    if isinstance(row.get("messages"), list) and row["messages"]:
        return tok.apply_chat_template(row["messages"], tokenize=False)
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return row["prompt"] + row["completion"]
    return None


def mean_ce(model, tok, texts, device):
    tot, n = 0.0, 0
    with __import__("torch").no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).to(device)
            inp = ids["input_ids"]
            if inp.shape[1] < 2:
                continue
            out = model(**ids, labels=inp)
            loss = float(out.loss)
            if loss != loss:  # NaN guard
                continue
            L = int(inp.shape[1]) - 1
            tot += loss * L
            n += L
    if n == 0:
        raise RuntimeError("no eval rows scored")
    return tot / n


def adapter_subfolder(adapter_repo):
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(adapter_repo)
    except Exception:
        return None
    if "final/adapter_model.safetensors" in files:
        return "final"
    ckpts = sorted(
        (f.split("/")[0] for f in files if f.startswith("checkpoint-") and f.endswith("/adapter_model.safetensors")),
        key=lambda s: int(s.split("-")[1]),
    )
    if ckpts:
        return ckpts[-1]
    if "adapter_model.safetensors" in files:
        return ""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "reward gate requires the training GPU"
    device = "cuda"
    torch.manual_seed(3407)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, total, source, has_heldout, slice_note = load_rows(args.dataset, MAX_ROWS)
    tok = AutoTokenizer.from_pretrained(args.base)
    texts = [render(r, tok) for r in rows]
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise RuntimeError("no renderable rows (messages or prompt/completion)")
    texts = texts[:MAX_ROWS]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    ce_base = mean_ce(model, tok, texts, device)

    sub = adapter_subfolder(args.adapter)
    if sub is None:
        raise RuntimeError("no adapter weights found in candidate repo (training incomplete)")
    from peft import PeftModel
    merged = PeftModel.from_pretrained(model, args.adapter, subfolder=sub) if sub else PeftModel.from_pretrained(model, args.adapter)
    merged.eval()
    ce_adapter = mean_ce(merged, tok, texts, device)

    delta = ce_base - ce_adapter
    rel = (delta / ce_base * 100.0) if ce_base else 0.0
    verdict = "IMPROVED" if delta > 0 else ("NO_GAIN" if delta == 0 else "REGRESSED")
    promotion = "RECOMMENDED" if (verdict == "IMPROVED" and rel >= 1.0) else "HELD"

    receipt = {
        "schema": SCHEMA,
        "run": args.run,
        "base": args.base,
        "adapter_repo": args.adapter,
        "adapter_subfolder": sub,
        "dataset": args.dataset,
        "dataset_file": source,
        "eval_rows": len(texts),
        "rows_total": total,
        "dedicated_heldout": has_heldout,
        "ce_base": round(ce_base, 6),
        "ce_adapter": round(ce_adapter, 6),
        "reward_delta": round(delta, 6),
        "relative_improvement_pct": round(rel, 3),
        "verdict": verdict,
        "promotion": promotion,
        "gpu": torch.cuda.get_device_name(0),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "Teacher-forced CE delta is a likelihood metric, not human quality.",
            "Eval slice: " + slice_note + ".",
            "Fail-closed: this receipt exists only because the full measurement ran.",
        ],
    }
    out = "/tmp/reward_receipt.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1)
    from huggingface_hub import HfApi
    HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
        path_or_fileobj=out,
        path_in_repo="rewards/" + args.run + "-reward.json",
        repo_id=OUT_REPO,
        repo_type="dataset",
    )
    print("REWARD_PUBLISHED", args.run, verdict, "delta", round(delta, 4))


if __name__ == "__main__":
    main()
