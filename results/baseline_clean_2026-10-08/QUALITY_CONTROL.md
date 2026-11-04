# QUALITY CONTROL — clean baseline (2026-10-08)

Automated by `tools/qc.py` → `qc.json`. Re-runs automatically when probing finishes
(`tools/finalize_when_done.py`). This document reflects the latest run.

## Current result: 13 / 13 PASS

| # | Check | Result |
|---|---|---|
| 1 | no invalid ExPLoRA embedding used | PASS |
| 2 | every canonical fine-tuned comparison has 3 seeds | PASS |
| 3 | full-train C selection (no subsample) | PASS |
| 4 | no 20,000-tile C-selection subsample | PASS |
| 5 | schema inferred from labels; n_train = 49,281 | PASS |
| 6 | RUBC retained at index 7 | PASS |
| 7 | test labels align with `splits/test_11cls.csv` | PASS |
| 8 | exactly 9 test orthomosaics | PASS |
| 9 | no train/test orthomosaic leakage | PASS |
| 10 | mono-thread BLAS recorded per probe | PASS |
| 11 | all outputs point to existing embedding paths | PASS |
| 12 | all used embeddings hash-recorded in the manifest | PASS |
| 13 | the five historical anchor files are unchanged (sha256) | PASS |

## Evidence

- Anchor sha256 (unchanged): `all_models_canonical_merged.json` `51b99254…`, `without_rhol/probe_knn_cgrid.json` `bc2fa06e…`, `significance_matrix_tier.json` `ad8b65f4…`, `tier_preds_cache.npz` `c8a3b00b…`, `splits/test_11cls.csv` `f06cac2c…`.
- Embedding manifest: 1,326 unique files / 27.27 GB hashed; 0 missing.
- Invalid-ExPLoRA exclusion verified by md5 (`d04cc317…` in all three invalid dirs).
- Valid LoRA r=8 verified distinct by md5 (`cbd8df15…`).

## Scope note

Checks 1, 3, 4, 8–13 are global and final. Checks 2, 5–7 are evaluated over the probes
that have completed; they are re-evaluated on the full set when the run finalizes.
