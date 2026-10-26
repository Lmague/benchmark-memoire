# Arctic-TVC vegetation benchmark (technical)

MSc project at UQAM: 42 vision foundation model configurations for 11-class tundra
vegetation mapping from drone orthomosaics. Headline: best macro F1 is 0.5080, reached
by adding spatial context at inference; fine-tuning a strong frozen backbone gains
0.012 or less on a single tile; the top tier cannot be ranked further at 9 test maps.

Every number below is traceable to a file in this repo.

## Protocol

| Item | Value |
|---|---|
| Data | Arctic-TVC, 38 drone orthomosaics at about 2.2 mm/px, Trail Valley Creek, NWT, 35 annotated. Public on FRDR: [doi:10.20383/103.01418](https://doi.org/10.20383/103.01418) |
| Split | Whole orthomosaic per side (no tile leakage). 9 test maps. Tiles: 49,433 train (12 classes), 49,281 train (11), 17,598 test |
| Classes | 12-class taxonomy. RHOL absent from val and test (0 tiles), so metrics use 11 classes. ARCA, DRYI, RUBC score near zero for every model |
| Readout | Multinomial logistic regression (lbfgs), shared grid C in {1e-4, 1e-3, 0.01, 0.1, 1, 10}, best C on val macro F1 |
| Models | 11 frozen + 31 fine-tuned x 3 seeds, all re-probed after training. Registry `results/all_models_canonical_merged.json`: 44 entries, 42 canonical (2 invalid ExPLoRA excluded) |
| Determinism | seed=42, single-threaded BLAS enforced (thread count moves F1 by about 0.0015) |
| Guards | `tests/test_repo_guards.py` checks the C grid, the thread policy, and deprecated values |
| Metric | Macro F1 on test, 11 classes |

## Experiments

| # | Experiment | Design | Headline result | Code | Data |
|---|---|---|---|---|---|
| 1 | Main ranking | 42 configs, one probe, one split | 0.5080 context fusion; 0.5057 same on SimDINOv2-B; 0.4870 context distillation; 0.4842 DINOv3-L LoRA; best frozen DINOv3-H+/16 0.4805 | `extract.py`, `probe.py`, `scripts/run_pipeline.py` stage A | `results/all_models_canonical_merged.json`, `figures/fig_readme_ranking.png` |
| 2 | Spatial context | Tile fused with 512/1024/2048 px neighbourhood, 5 frozen backbones | +0.025 to +0.034, positive on 8 of 9 maps; 512 beats 1024 beats 2048; frozen SimDINOv2-B with fused readout 0.5059, no training | `scripts/context_crop.py`, `context_size_sweep.py`, `context_fused_probe_controls.py`, 11 x `scripts/slurm_context_*.sh` | `figures/fig_readme_context.png` |
| 3 | Context controls | Tile alone vs context alone vs shuffled context | 0.4778 / 0.4781 / 0.4745. The aligned neighbourhood carries the signal, extra pixels do not | `scripts/context_bouguessa_controls.py` | same figure |
| 4 | Context distillation | Neighbourhood knowledge distilled into a tile-only embedding | 0.4870 best single-tile model | `configs/context_distill_dinov3b.yaml`, `context_distill_simdinov2b.yaml`, `scripts/context_distill*.py` | `results/all_models_canonical_merged.json` |
| 5 | PEFT regimes | LoRA r=8, full FT, MHSA, norm-only on DINOv3-B / SimDINOv2-B / ViT-B | LoRA +0.0114 over its own frozen baseline; full FT of ViT-B 0.4786, still below frozen ViT-L 0.4792; norm+head (47k params) within 0.001 of LoRA r=8 | `train.py`, `configs/ft_*.yaml`, `*_lora*.yaml` | `results/bootstrap_stageA_paired_CANONICAL.json`, `results/simb_stageA_probe_CANONICAL.json` |
| 6 | LoRA ablation | Rank 2/4/8/16/32, scaling, blocks 0-5/6-11/9-11, Q+K+V targets | all variants within 0.0067; higher scaling is worse | `configs/dinov3_vitb16_lvd_lora_*.yaml`, `configs/simdinov2_vitb16_lora_*.yaml` | `results/lora_simb_ablation/config_used_*.yaml`, `results/rapport_data/tile_heads/` |
| 7 | Readout heads | Linear probe vs k-NN, MLP-2, bilinear, FiLM. 33 tile groups, 24 fused tags | Non-linear heads never beat the canonical lbfgs probe: 0/33 tile (mean -0.0103, max -0.0010), 2/24 fused (max +0.0076, seed-unique tags, not citable). k-NN 0.031 to 0.104 below. MLP-2 beats an AdamW baseline only | `scripts/tile_head_sweep.py`, `scripts/fusion_head_sweep.py`, `scripts/rapport/head_sweep_summary.py` | `results/rapport_data/head_sweep_summary.{json,csv}`, `results/rapport_data/tile_heads/`, `knn_vs_probe.csv` |
| 8 | Latent geometry, model ranking | 14 measures (Fisher, CH, DB, NC1, silhouette, RankMe, effective rank, participation, stable rank, intrinsic dim, anisotropy, k-NN purity, LogME) vs F1, leave-one-model-out on the 35-model tier (1,184 pairs) | Separability generalizes: 63.5 to 69.4 % correct order out-of-sample, p(BH) <= 1.3e-4. Spectral does not: RankMe 37.2 %, LogME-train 33.6 %, unstable sign. On the full population silhouette rho = 0.825 (n=12) | `scripts/task_a_spectrum.py`, `scripts/rapport/geometry_test_fixed_n.py`, `scripts/rapport/loo_triage.py` | `results/rapport_data/loo_triage.{json,csv}`, `results/geometry_extended_12models.json`, `results/rapport_data/correlations_geometry.csv`, `figures/fig_readme_latent.png` |
| 9 | Layerwise geometry | Same measures across the 12 layers of one backbone | Within one model, geometry ranks its own layers: RankMe rho = 0.89 vs F1, anisotropy rho = -0.92 | `scripts/rapport/layerwise_geometry.py` | `results/rapport_data/layerwise.csv` |
| 10 | Dimension probe | PCA truncation k = 10 to full, canonical probe, 5 models | 100 components give 99.1 % of full F1 on DINOv3-B (668 remaining dims add 0.004); 97.4 to 100.3 % across models. Diagnostic probe on reduced C grid, values slightly below canonical | `scripts/rapport/dim_probe.py`, `dim_probe_summary.py` | `results/rapport_data/dim_probe_summary.csv`, `dim_probe_*.json` |
| 11 | Texture features | 106 handcrafted descriptors, 7 families (GLCM, GLRLM, GLSZM, GLDM, NGTDM, DWT, Fourier), fused with embeddings | Probe gain -0.002 to +0.001; embeddings already predict texture (R2 above 0.88) | `src/texture.py`, `scripts/texture_ablation.py`, `texture_redundancy.py`, `scripts/slurm_texture.sh` | `results/texture/ablation_*.json`, `redundance_*.json` |
| 12 | LogME | Closed-form transferability estimate vs F1 | Bimodal, cannot rank models; fails LOO (row 8) | `scripts/task_b_logme.py` | `results/rapport_data/loo_triage.csv` |
| 13 | Learning curves | Frozen baseline vs LoRA / full / MHSA, 1 % to 100 % of training tiles | LoRA needs about 4,592 tiles to match its own frozen baseline and ends +0.012 above; weak ImageNet regimes overtake at about 2,060 tiles, +0.025 to +0.028 | `scripts/datacurve_one_run.py`, `datacurve_aggregate.py`, `scripts/slurm_datacurve*.sh`, `configs/*datacurve*.yaml` | `figures/fig_readme_datacurve.png`, `results/rapport_data/geometry_datacurve.csv` |
| 14 | Sampling control | Random tile draw vs spatially blocked draw at low fractions | At 1 % of data: 0.426 random vs 0.099 blocked. Only 3 to 6 classes present at 1 %; scoring present classes only gives 0.229. The drop is class coverage first, spatial leakage second | same scripts | `figures/fig_readme_datacurve.png` |
| 15 | Pretraining domain | Satellite vs natural-image pretraining, same architectures | SatMAE 0.4091, Scale-MAE 0.4480, DINOv3-L satellite 0.4620 vs 0.4792 DINOv3-L on natural images | `configs/frozen_*.yaml` | `results/all_models_canonical_merged.json` |
| 16 | From scratch | ViT-B/16 trained on Arctic-TVC only | 0.3561, last of 42 | `configs/vitb16_scratch*.yaml`, `train.py` | same registry |
| 17 | Significance | Paired per-orthomosaic bootstrap on the competitive tier: 35 groups, 595 pairs, n=10,000, BH alpha 0.05 | 182 rejections; marginal model CI 0.0142 vs 0.1788 with cluster bootstrap over maps; smallest detectable effect about 0.03 to 0.05. Control pair ViT-B/16 vs its FT: delta -0.0076, p 0.003 | `scripts/significance_matrix.py`, `scripts/rapport/significance_tier.py`, `scripts/task_c_aso.py`, `task_c_b_paired.py`, `bootstrap_ci.py` | `results/significance_matrix_tier.json`, `significance_matrix_all12.{json,png}`, `T2c_headline_pair_p_CANONICAL.json`, `T2a_kmeans_control.json`, `silhouette_robustness_CANONICAL.json`, `bootstrap_ci_2026-07-09.json` |
| 18 | Class-schema diagnostics | Same probe with and without RHOL, plus 8-class scheme | Reprobe differs from the canonical campaign by <= 0.0003 on 5 models; RHOL has 0 tiles in val and test | `scripts/generate_splits_11cls.py`, `scripts/rapport/registry.py` | `results/with_rhol/probe_knn_cgrid.json`, `results/without_rhol/probe_knn_cgrid.json` |
| 19 | Per-class analysis | Per-class F1 for all models, per regime and per tier; confusion matrices | ARCA, DRYI, RUBC near zero for every model everywhere | `scripts/rapport/make_perclass_tier.py`, `scripts/rapport/knn_and_confusion.py`, `scripts/analyze_latent_classes.py` | `results/rapport_data/per_class_*.csv`, `confusion_tier.npz`, `confusion_matrices.npz` |
| 20 | Artifact consistency | 15 machine checks: registry vs published F1, tile counts, deprecated values, table balance, figure references | 15/15 pass; run-vs-canonical probe deviation max 0.0043 | `scripts/rapport/check_consistency.py` | `results/rapport_data/consistency_report.md` |

Ongoing, not citable yet: `leo-explora-ssl/` (27 files) holds the self-supervised
pretraining exploration on Arctic imagery with SLURM pipelines; its outputs
(`results/leo_explora_ssl/`, 15 GB) are not published here.

## Limits and withdrawals

- The tier cannot be ranked at 9 test maps. An earlier version of this README said no
  pairwise difference survived correction: that was a flaw in the test, not a property
  of the data (bootstrap p resolution 0.0002 vs the 0.00008 needed at the top rank).
  Treat differences below about 0.03 as undetectable, and 0.005-level claims as not final.
- Per-seed choice of C moves some models: DINOv3-L LoRA reads 0.4842 or 0.4799
  depending on which C is kept.
- Deprecated values 0.4675, 0.4680 and 0.5256 must not be cited; `tests/test_repo_guards.py`
  enforces their absence. The single-seed fine-tuned models (`resnet50_arctic`,
  `vitb16_arctic`, `vitb16_fulft_arctic`) and the invalid ExPLoRA runs are excluded from
  all canonical tables.
- Macro F1 is dragged down by 3 classes that no model can learn at this resolution.

## Repository layout

```
src/            engine, models, probe, features, latent, texture, transfer... (14 modules)
configs/        76 YAMLs: frozen_*, ft_*, *_lora_*, context_*, *datacurve*
scripts/        111 files: run_pipeline.py (stages A-F), context_*, datacurve_*,
                texture_*, task_a..d, significance*, slurm_* (36), rapport/ (~30)
probe.py        CLI: linear probe + k-NN          extract.py  CLI: feature extraction
analyze.py      CLI: latent geometry              train.py    CLI: fine-tuning
make_figures.py CLI: paper figures                scripts/make_readme_figures.py  README figures
tests/          repo guards, 11-class split, texture contract
results/        authoritative JSON/CSV outputs (one file per published number)
figures/        README + report figures           leo-explora-ssl/  SSL exploration (SLURM)
```

## Reproduce

```bash
pip install -r requirements.txt
# DINOv3 weights are gated: HF access to facebook/dinov3-*-lvd1689m.
# SimDINOv2 needs: pip install git+https://github.com/ilyassmoummad/sslplant

python scripts/generate_splits_11cls.py        # splits are not included
python extract.py --config configs/frozen_dinov3_lvd.yaml
python probe.py   --config configs/frozen_eval.yaml
python analyze.py --config configs/frozen_eval.yaml

make paper        # full pipeline A-F, deterministic, seed=42
make probe | latent | sig | transfer | figures | tables   # single stages
```

Two rules make the numbers canonical, and `tests/test_repo_guards.py` checks both:
single-threaded BLAS before any numpy import (thread count changes lbfgs, worth about
0.0015 F1), and one shared 6-point C grid for every model.

Embeddings and splits are not in the repo (size). Arctic-TVC is public on FRDR.
Code MIT, figures and results CC-BY-4.0.
