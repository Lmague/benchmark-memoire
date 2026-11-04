# Statistical Protocol — Frozen

Status: **FROZEN / NORMATIVE**. This document defines the statistical analysis protocol for the baseline. It distinguishes **confirmatory** (primary BH family) from **exploratory** analyses.

---

## PRIMARY UNIT

- **PRIMARY UNIT = orthomosaic.**
- There are **9 independent test orthomosaics**.
- The ortho is derived from the test CSV filepath component **index 1** (pattern `arctic_vegetation/<ortho>/<CLASS>/tile`).
- `n_tiles` = **17,598**.

---

## PRIMARY analysis

- **Paired bootstrap resampling over the 9 orthomosaics** (cluster bootstrap) **AND**
- **Hierarchical bootstrap** (resample orthomosaics with replacement, then resample tiles within each drawn orthomosaic with replacement).
- `n` = **10,000**, `seed` = **42**.
- **Shared resample indices across all compared models and seeds (paired).**

---

## SECONDARY

- **Tile-level paired bootstrap** is retained **only as a lower-bound analysis**.

---

## Per-comparison reporting (required fields)

For **every comparison** report:

- **n independent units** = **9**
- **effect size** (**Delta F1**)
- **95% CI**
- **CI width**
- **p-value**
- **BH-adjusted q-value**
- **whether it survives BH at alpha=0.05**

| Field | Value / descriptor |
|---|---|
| n independent units | 9 |
| Effect size | Delta F1 |
| 95% CI | reported |
| CI width | reported |
| p-value | reported |
| BH-adjusted q-value | reported |
| Survives BH at alpha=0.05 | reported (yes/no) |

---

## PRIMARY BH FAMILY (frozen a priori)

- **All pairwise comparisons among the INCLUDED canonical models in the competitive tier** (all pairs of included model keys with `>=1` seed).
- **Pairs are frozen before results are examined.**
- **Supporting/datacurve models are exploratory and are NOT in the primary BH family.**

---

## Single-seed models

- **Single-seed models (frozen backbones)** are reported as **point estimates** and are **excluded from seed-paired tests**.

---

## Confirmatory vs exploratory

- **Confirmatory**: primary family, **BH-corrected**.
- **Exploratory**: everything else.
- **Do not claim significance from descriptive-only evidence.**
