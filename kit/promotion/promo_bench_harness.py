"""promo_bench_harness.py -- scaled promotion-evidence bench for frontier adapters.

Mirrors reward_gate_v1_1 methodology (same file pickers, row rendering,
teacher-forced CE, adapter subfolder resolution) but at larger row counts,
producing szl.bench/v1 receipts as OWNER-GATED promotion evidence in
SZLHOLDINGS/wave-logs bench/promo-evidence/. Evidence only: no receipt here
authorizes any promotion; every promotion decision stays owner-gated.
Fail-closed: any error means no receipt.
"""
import argparse
import datetime as dt
import json

SCHEMA = "szl.bench/v1"
OUT_REPO = "SZLHOLDINGS/wave-logs"
OUT_DIR = "bench/promo-evidence"
BENCH_VERSION = "v1-20261119"

EVAL_PREFS = [
    "heldout.jsonl", "eval.jsonl", "test.jsonl", "validation.jsonl",
    "data/heldout.jsonl", "data/eval.jsonl", "data/test.jsonl",
]
TRAIN_PREFS = ["train.jsonl", "data/train.jsonl"]
TRAIN_PREFS_JSON = ["data/train.json", "train.json"]


def pick_file(files, prefs):
    for p in prefs:
        if p in files:
            return p
    return None


def load_rows(dataset, max_rows):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(dataset, repo_type="dataset")
    jsonls = sorted(f for f in files if f.endswith(".jsonl"))
    heldout = pick_file(files, EVAL_PREFS)
    json_array_mode = False
    if jsonls:
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
    else:
        single_jsons = sorted(f for f in files if f.endswith(".json"))
        source = pick_file(files, TRAIN_PREFS_JSON) or (single_jsons[0] if single_jsons else None)
        if source is None:
            raise RuntimeError("dataset has no jsonl or json data files")
        path = hf_hub_download(dataset, source, repo_type="dataset")
        payload = json.load(open(path, encoding="utf-8", errors="replace"))
        if isinstance(payload, dict):
            lists = [v for v in payload.values() if isinstance(v, list)]
            payload = lists[0] if lists else None
        if not isinstance(payload, list):
            raise RuntimeError("json data file is not a row array")
        rows = payload
        json_array_mode = True
    total = len(rows)
    if heldout is not None:
        slc = rows[:max_rows]
        note = "dedicated eval file"
    else:
        slc = rows[-max_rows:]
        note = "dataset tail slice (may overlap training data)"
    if json_array_mode:
        note = note + " [single-file JSON array]"
    return slc, total, source, heldout is not None, note


def render(row, tok):
    if isinstance(row.get("messages"), list) and row["messages"]:
        return tok.apply_chat_template(row["messages"], tokenize=False)
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return row["prompt"] + row["completion"]
    if isinstance(row.get("text"), str) and row["text"].strip():
        return row["text"]
    return None


def mean_ce(model, tok, texts, device):
    tot, n = 0.0, 0
    import torch
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=1024).to(device)
            inp = ids["input_ids"]
            if inp.shape[1] < 2:
                continue
            out = model(**ids, labels=inp)
            loss = float(out.loss)
            if loss != loss:
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
    ap.add_argument("--rows", type=int, default=500)
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "promo bench requires a GPU"
    device = "cuda"
    torch.manual_seed(3407)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, total, source, has_heldout, slice_note = load_rows(args.dataset, args.rows)
    tok = AutoTokenizer.from_pretrained(args.base)
    texts = [render(r, tok) for r in rows]
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise RuntimeError("no renderable rows")
    texts = texts[: args.rows]

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    ce_base = mean_ce(model, tok, texts, device)

    sub = adapter_subfolder(args.adapter)
    if sub is None:
        raise RuntimeError("no adapter weights found in candidate repo")
    from peft import PeftModel
    merged = PeftModel.from_pretrained(model, args.adapter, subfolder=sub) if sub else PeftModel.from_pretrained(model, args.adapter)
    merged.eval()
    ce_adapter = mean_ce(merged, tok, texts, device)

    delta = ce_base - ce_adapter
    rel = (delta / ce_base * 100.0) if ce_base else 0.0
    verdict = "IMPROVED" if delta > 0 else ("NO_GAIN" if delta == 0 else "REGRESSED")

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
        "evidence_only": True,
        "owner_gate": "Promotion is owner-gated; this bench receipt authorizes nothing by itself.",
        "gpu": torch.cuda.get_device_name(0),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": [
            "Scaled promotion-evidence bench " + BENCH_VERSION + "; mirrors reward_gate_v1_1 methodology at higher row counts.",
            "Eval slice: " + slice_note + ".",
            "Teacher-forced CE delta is a likelihood metric, not human quality.",
            "Fail-closed: this receipt exists only because the full measurement ran.",
        ],
    }
    import tempfile, os
    from huggingface_hub import HfApi
    fd, tmp = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    HfApi().upload_file(path_or_fileobj=tmp, path_in_repo=OUT_DIR + "/" + args.run + "-bench.json", repo_id=OUT_REPO, repo_type="dataset")
    os.unlink(tmp)
    print("BENCH_OK", args.run, "rows", len(texts), "rel_pct", round(rel, 3), "verdict", verdict)


if __name__ == "__main__":
    main()
