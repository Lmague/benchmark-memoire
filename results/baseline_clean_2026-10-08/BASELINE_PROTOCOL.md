# Baseline Protocol — Frozen

Status: **FROZEN / NORMATIVE**. This document defines the baseline evaluation protocol. It is a frozen specification of an already-decided protocol; it is descriptive and normative, not an experiment to be revised.

---

## DATA

- Dataset: **Arctic-TVC**.
- Split granularity: **existing orthomosaic-level split**.
- Test = **17,598 tiles** over **9 independent test orthomosaics**.
- Split files:
  - `splits/{train,val,test}.csv` — **12-class**; train = **49,433**.
  - `splits/{train,val,test}_11cls.csv` — **11-class**; train = **49,281**; val = **13,209**; test = **17,598**.
- Class order (12): **ALDE ARCA BIRC DRYI LICH MOSS PETF RHOL RUBC SEDG TUSS WILL**.

---

## SCHEMA

- Reported schema = **11 classes**.
- **RHOL removed**.
- **RUBC remains class index 7**.
- Source numbering is **INFERRED FROM ACTUAL LABELS** (ymax over train+val+test concatenated: `>=11 => 12cls`, else `11cls`), **NEVER from model names**.
- **Never apply a blind "drop index 7" to an 11-class embedding.**

---

## RHOL handling

- For **12-class-numbered sources**: **RHOL-labelled SAMPLES are removed** from train/val/test and **indices >7 are recompacted by -1** (this matches the 49,281 / 13,209 / 17,598 11-class split).
- For **11-class sources**: **no removal is applied**.

---

## PROBE

- `StandardScaler` **fit on train only** (transform val/test).
- `LogisticRegression`:
  - solver = `lbfgs`
  - multinomial (on sklearn `>=1.8` `multi_class` is removed and `lbfgs` is intrinsically multinomial; pass `multi_class` only where supported)
  - `max_iter` = **2000**
  - `random_state` = **42**
  - `C` grid = **[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]**

---

## C SELECTION

- Select `best_C` on the **FULL training set validation procedure**.
- The **20,000-tile subsample** (`CGRID_SUBSAMPLE` / `cgrid_subsample`) is **FORBIDDEN** in this baseline.
- Record the validation score for **every C**, the **selected C**, and the **validation margin between best and second-best**.
- **Deterministic tie-break rule**: on an exact tie of validation `f1_macro_all`, choose the **LARGER C**.

---

## FINAL FIT

- Refit on the **complete train set** using the **selected C**.

---

## TEST METRICS

- Report:
  - `f1_macro_pres` — macro-F1 over classes **present in `y_true`** — the **PRIMARY reported metric**.
  - `f1_macro_all` — macro-F1 over `labels=range(11)`.
- Model **SELECTION** uses `f1_macro_all` (as in the historical canonical protocol); **REPORTING** uses `f1_macro_pres`.
- **State this distinction explicitly.**

---

## THREADING

- **Before importing numpy/sklearn** set:
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
  - `VECLIB_MAXIMUM_THREADS=1`

---

## SEEDS

- Every **fine-tuned comparative claim requires 3 seeds**.
- Report **mean +/- SD** with `ddof=1`.
- **Never majority-vote seeds**.
- Single-seed values are labelled **point estimates**.

---

## EXCLUSIONS

- Exclude invalid ExPLoRA embeddings:
  - `embeddings/dinov3_vitb16_lvd_explora_*`
  - `DINOv3_ExPLoRA_Like/*`
  - `dinov3_vitb16_lvd_explora/*`
- **KEEP** the valid **LoRA r=8** at `DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed*` (name contains "explora" but it is valid LoRA r8).
- Selection is by **explicit path map** (`model_map.json`), **never by filename**.
- Deprecated seed-unique historical FT (`resnet50_arctic`, `vitb16_arctic`, `vitb16_fulft_arctic`) are **excluded from comparative claims**; their artefacts are **preserved untouched**.

---

## PROVENANCE

Every probe records:

- embedding **sha256 hashes** (from `embedding_manifest.json`),
- **git revision**,
- **sklearn/numpy versions**,
- **runtime**.
