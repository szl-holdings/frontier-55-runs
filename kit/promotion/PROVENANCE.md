# Provenance — promotion gate kit mirror

Mirrored 2026-11-24 under owner green light ("GREEN LIGHT GET IT ALL DONE NOW"; standing directive: all models coded back into the szl-holdings org, pure Python).

| File | Bytes | Canonical source | Fetched |
|---|---|---|---|
| gate_promotion.py | 17828 | hf://models/SZLHOLDINGS/frontier-promotion-gate-v1/gate_promotion.py (private model repo, same file the promotion-gate model card describes) | 2026-11-24, byte-exact via HF API |
| gate_promotion.selftest_receipt.json | 182 | same repo | 2026-11-24 |
| reward_gate.py | 6966 | hf://datasets/SZLHOLDINGS/wave-logs/kit/reward-gate/reward_gate.py (kit re-hosted from w29-runs per wave-logs kit/manifest.json pin df5d9c1169c7) | 2026-11-24 |
| promo_bench_harness.py | 7778 | hf://datasets/SZLHOLDINGS/wave-logs/kit/bench/promo_bench_harness.py (same kit pin) | 2026-11-24 |

Honesty notes:
- Bytes are mirrored unmodified; no metric, threshold, or logic was edited by this commit. The source model repo is private; this commit mirrors eval-gate CODE only (no weights, no candidate data, no secrets) so the org carries the promotion-gate Python per the owner directive. Nothing here lifts the promotion HOLD; the countersign scope (listing-only, lanes parked, replicas 0) is unchanged.
- Map note: this mirror is the evidence basis for the szl-asset-registry intake row proposal SZLHOLDINGS/frontier-promotion-gate-v1 -> frontier-55-runs (data/intake_new_models_20261124.json, merged PR #34 2026-11-24). Any override mapping stays with the owning tools + owner adjudication (fail-closed).
- Any file listed SKIPPED was not fetched at commit time; nothing was fabricated to fill its place.