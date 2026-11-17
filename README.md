# frontier-55-runs

**The GitHub home of the SZL Holdings flagship-55 frontier run evidence.**

Owner directive, 17 Nov 2026: the frontier evidence goes in GitHub, fully Python, public. The Hugging Face dataset [SZLHOLDINGS/frontier-55-runs](https://huggingface.co/datasets/SZLHOLDINGS/frontier-55-runs) is the write-locked system of record for run data; this repo is the protocol-facing public push of everything the collaboration needs.

## Status (measured 2026-11-17T19:28Z)

- Training lane szl-train-flagship55-v1 observed booting (Salad census).
- Companion dataset [SZLHOLDINGS/flagship55-sft-v1](https://huggingface.co/datasets/SZLHOLDINGS/flagship55-sft-v1) created 2026-11-17T19:02Z.
- No run evidence has landed yet; this repo will mirror the public HF evidence byte-exactly (anonymous fetch, sizes verified against the HF tree API) as it arrives.

## Honesty notes

- Nothing in this repo claims a metric not present in the receipts.
- Private artifacts and evaluation decisions stay in the system of record; this repo publishes public evidence only.
- Companion kit: [vgl-cold-replay](https://github.com/szl-holdings/vgl-cold-replay) (fully Python, stdlib-only, anonymous-only).

## Governance

- Receipts are hash-chained, MEASURED-energy, fail-closed. See [governed-receipt-spec](https://github.com/szl-holdings/governed-receipt-spec).
- Monetization lane: governed inference + x402 payment gating on Cloudflare (inference.a-11-oy.com).
