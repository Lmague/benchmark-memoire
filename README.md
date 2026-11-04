# Arctic-TVC: foundation models for tundra vegetation mapping

MSc project in computer science (AI), UQAM. Supervisors: Mohamed Bouguessa (ML) and Etienne Laliberté (ecology).

The task is to label 224 px tiles cut from drone orthomosaics of arctic tundra (about 2.2 mm/px) into 11 vegetation classes. I use frozen or lightly adapted vision foundation models, one shared linear probe and one test split, and ask what actually improves the map: a better backbone, fine-tuning, or the tiles around each tile.

What I found:

- Among DINOv3 and SimDINOv2 backbones, size and adaptation method move macro F1 by 0.015 at most (0.004 to 0.012 depending on the comparison). At 9 test maps none of these differences is separable from the best frozen model. Only the weakest backbones (SatMAE, ResNet-50 ImageNet, ViT-B/16 from scratch) are.
- Using neighbouring tiles moves F1 by 0.03 to 0.06, and it needs no training.
- Learned aggregation of neighbours (attention, GATv2) does not beat a plain average.
- Embedding geometry can rank models somewhat better than chance, but it does not predict F1.

Best result: **0.5359 macro F1** with a frozen SimDINOv2-B/16, a 3x3 average of tile embeddings and mean-field smoothing of the predicted map (smoothing strength and iterations chosen on validation). The same backbone on a single tile reads 0.4719, so the gain is +0.064. On the 8 reliable classes the same recipe reads 0.7369 (see [Per class](#per-class)). This recipe reads neighbouring tiles at test time, so compare it with other models that do (0.505 to 0.521, from two different runs), not with the single-tile ranking (best 0.4872).

Contents: [Ranking](#ranking-of-42-configurations) · [Adaptation](#adaptation-methods) · [Context and smoothing](#neighbouring-tiles-and-smoothing) · [Learned aggregation](#learned-aggregation) · [Embedding geometry](#embedding-geometry) · [Data size](#data-size) · [Per class](#per-class) · [Protocol](#protocol) · [Ongoing](#ongoing-work) · [Limits](#limits) · [Reproduce](#reproduce)

## Ranking of 42 configurations

![Named ranking and significance](figures/readme/01_leaderboard.png)

11 frozen models and 31 fine-tuned ones (3 seeds each), all read with the same linear probe. The figure has three levels of input: single tile (top 18 models plus 7 reference points), neighbour tiles as input, and smoothing of the predicted map. The full list is below.

- Best frozen model: DINOv3 ViT-H+/16, 0.4805. Best plain LoRA in this ranking: DINOv3-L, 0.4842. That is +0.0037, CI [-0.034, +0.017], q = 0.76: not separable.
- Only 3 of 38 single-tile models separate from the best frozen model after Benjamini-Hochberg correction: SatMAE (-0.071), frozen ResNet-50 ImageNet (-0.073) and ViT-B/16 from scratch (-0.126). Everything else overlaps with it at 9 maps.
- Satellite pretraining does not look better than natural images, but the data cannot say more. DINOv3 ViT-L on satellite images reads 0.4620 against 0.4791 for the natural-image version (-0.017, CI [-0.058, +0.001], not separable). Scale-MAE (0.4478) and SatMAE (0.4097) sit below frozen ViT-B/16 ImageNet (0.4497).
- Fine-tuning on top of the same 512 px fusion does not visibly help: LoRA variants read 0.5056 and 0.5052 (clean baseline) against 0.5058 for the frozen fusion (tile-context run). Different runs, so this is indicative only.
- Neighbour-aware models (0.5052 to 0.5205, clean baseline and tile-context run) sit above every single-tile model. They are outside the test family above, so no test against the best frozen model is reported for them here.
- Non-linear readouts (kNN, MLP, bilinear, FiLM) never beat the linear probe on tile features (0 of 33 groups, mean delta -0.0103). They do on 2 of 24 fused groups.

<details>
<summary>All 42 configurations</summary>

| # | Model | Macro F1, 11 classes | Macro F1, 8 classes | Seed SD (11 / 8) | Runs |
|---:|---|---:|---:|---:|---:|
| 1 | Context fusion, fine-tuned | 0.5079 | 0.6984 | 0.0015 / 0.0020 | 3 |
| 2 | Context fusion, SimDINOv2-B LoRA r8a16 | 0.5056 | 0.6953 | 0.0071 / 0.0098 | 3 |
| 3 | Context fusion, SimDINOv2-B LoRA r2a4 | 0.5052 | 0.6947 | 0.0030 / 0.0042 | 3 |
| 4 | Context distillation, tile only at inference | 0.4872 | 0.6699 | 0.0018 / 0.0025 | 3 |
| 5 | DINOv3-L, LoRA r=8 | 0.4842 | 0.6657 | 0.0012 / 0.0017 | 3 |
| 6 | Context distillation, EMA teacher, tile only | 0.4835 | 0.6648 | 0.0017 / 0.0024 | 3 |
| 7 | SimDINOv2-L, LoRA r=8 | 0.4835 | 0.6633 | 0.0068 / 0.0106 | 3 |
| 8 | DINOv3-B, LoRA r=8 | 0.4830 | 0.6641 | 0.0037 / 0.0051 | 3 |
| 9 | SimDINOv2-B, LoRA r=8, Q+K+V | 0.4812 | 0.6617 | 0.0052 / 0.0072 | 3 |
| 10 | SimDINOv2-B, LoRA r=8, blocks 9-11 | 0.4805 | 0.6608 | 0.0001 / 0.0001 | 3 |
| 11 | DINOv3 ViT-H+/16, frozen | 0.4805 | 0.6606 | n/a / n/a | 1 |
| 12 | SimDINOv2-B, LoRA r=8, blocks 6-11 | 0.4803 | 0.6604 | 0.0028 / 0.0038 | 3 |
| 13 | SimDINOv2-B, LoRA r=2 | 0.4793 | 0.6591 | 0.0024 / 0.0033 | 3 |
| 14 | DINOv3 ViT-L/16, frozen | 0.4791 | 0.6588 | n/a / n/a | 1 |
| 15 | DINOv3-B, attention only | 0.4790 | 0.6572 | 0.0025 / 0.0047 | 3 |
| 16 | DINOv3-B, full fine-tuning | 0.4788 | 0.6535 | 0.0025 / 0.0032 | 3 |
| 17 | SimDINOv2-B, LoRA r=4 | 0.4783 | 0.6577 | 0.0050 / 0.0069 | 3 |
| 18 | SimDINOv2-B, norm layers + head only | 0.4783 | 0.6577 | 0.0002 / 0.0003 | 3 |
| 19 | SimDINOv2-B, full fine-tuning | 0.4782 | 0.6575 | 0.0074 / 0.0102 | 3 |
| 20 | SimDINOv2-B, LoRA r=8 | 0.4780 | 0.6572 | 0.0029 / 0.0039 | 3 |
| 21 | DINOv3-S, LoRA r=8 | 0.4776 | 0.6567 | 0.0023 / 0.0031 | 3 |
| 22 | SimDINOv2-B, LoRA r=8, blocks 0-5 | 0.4771 | 0.6560 | 0.0023 / 0.0032 | 3 |
| 23 | SimDINOv2-B, LoRA r=16 | 0.4769 | 0.6557 | 0.0015 / 0.0021 | 3 |
| 24 | SimDINOv2-B, LoRA r=8, scaling 2 | 0.4761 | 0.6547 | 0.0017 / 0.0023 | 3 |
| 25 | SimDINOv2 ViT-L/16, frozen | 0.4756 | 0.6540 | n/a / n/a | 1 |
| 26 | SimDINOv2-B, LoRA r=8, rsLoRA | 0.4751 | 0.6533 | 0.0020 / 0.0027 | 3 |
| 27 | SimDINOv2-B, LoRA r=32 | 0.4751 | 0.6532 | 0.0018 / 0.0025 | 3 |
| 28 | SimDINOv2-B, LoRA r=16, scaling 2 | 0.4750 | 0.6531 | 0.0026 / 0.0036 | 3 |
| 29 | ViT-B/16 ImageNet, full fine-tuning | 0.4748 | 0.6513 | 0.0015 / 0.0016 | 3 |
| 30 | SimDINOv2-B, LoRA r=16, rsLoRA | 0.4735 | 0.6497 | 0.0035 / 0.0034 | 3 |
| 31 | SimDINOv2 ViT-B/16, frozen | 0.4719 | 0.6488 | n/a / n/a | 1 |
| 32 | DINOv3 ViT-B/16, frozen | 0.4715 | 0.6483 | n/a / n/a | 1 |
| 33 | SimDINOv2-B, attention only (MHSA) | 0.4692 | 0.6430 | 0.0022 / 0.0031 | 3 |
| 34 | DINOv3 ViT-S/16, frozen | 0.4689 | 0.6448 | n/a / n/a | 1 |
| 35 | ViT-B/16 ImageNet, attention only | 0.4686 | 0.6430 | 0.0048 / 0.0082 | 3 |
| 36 | DINOv3 ViT-L/16 satellite, frozen | 0.4620 | 0.6352 | n/a / n/a | 1 |
| 37 | ResNet-50 ImageNet, full fine-tuning | 0.4573 | 0.6287 | 0.0033 / 0.0046 | 3 |
| 38 | ViT-B/16 ImageNet, frozen | 0.4497 | 0.6163 | n/a / n/a | 1 |
| 39 | Scale-MAE ViT-L/16, frozen | 0.4478 | 0.6157 | n/a / n/a | 1 |
| 40 | SatMAE ViT-L/16, frozen | 0.4097 | 0.5633 | n/a / n/a | 1 |
| 41 | ResNet-50 ImageNet, frozen | 0.4077 | 0.5606 | n/a / n/a | 1 |
| 42 | ViT-B/16 from scratch | 0.3547 | 0.4805 | 0.0115 / 0.0067 | 3 |

</details>

## Adaptation methods

![Adaptation regimes](figures/readme/02_adaptation_regimes.png)

16 ways to adapt SimDINOv2-B/16 end between 0.4692 and 0.4812. Frozen is 0.4719. None of the 16 separates from it (best: LoRA r=8 on Q+K+V, +0.0094, CI [-0.006, +0.020], q = 0.24).

- LoRA rank 2 to 32 spans 0.4793 to 0.4751 and slowly decreases with rank.
- Training only the norm layers and the head (0.4783) matches LoRA r=8 (0.4780) and full fine-tuning (0.4782).
- Full fine-tuning has the largest seed spread: 0.4696 to 0.4827 over 3 seeds.
- Attention-only tuning ends below frozen (0.4692).

## Neighbouring tiles and smoothing

![Context ladder](figures/readme/03_context_ladder.png)

All rows use the same frozen SimDINOv2-B/16 embeddings and the canonical probe. Deltas are paired, bootstrapped over the 9 test orthomosaics, 10,000 replicates.

- Adding context tiles raises F1 from 0.4718 to 0.5205. The step from tile + 512 px context to the 3x3 mean (+0.0136) is not separable from zero at 9 test maps.
- Mean-field smoothing of the predicted map is the clearest single gain: +0.0578 on a single tile, +0.0152 on top of the 3x3 mean. Both intervals exclude zero.
- After smoothing, the 3x3 mean adds little over a single tile: +0.0062, CI [-0.010, +0.053].
- Smoothing has no learned parameters and no training. Its strength and number of iterations (5 or fewer) are chosen on validation. Neighbour embeddings come from a cache.
- Colour and texture descriptors concatenated to the embedding lower F1 by about 0.004.

![Where smoothing acts](figures/readme/04_smoothing.png)

- 301 of the 349 net corrected tiles sit in mixed neighbourhoods (3 to 5 of 8 neighbours share the label). Tiles on true boundaries lose (-55).
- The nearest ring works best (tiles are 224 px, stride 112, so ±112 px is 50% overlap): 0.5294, against 0.5102 at ±224 px and 0.4957 at ±336 px. Combining rings (0.5195 to 0.5228) does not beat the nearest ring alone.

## Learned aggregation

![Learned aggregation](figures/readme/05_learned_aggregation.png)

- Attention over the 3x3 window reads 0.5085 (seeds 0.4928 to 0.5204) against 0.5205 for the mean: -0.012, CI [-0.044, +0.002].
- Shuffling the neighbour positions does not hurt attention (0.5115). The spatial layout is not used.
- GATv2 matches a uniform mean on the same graph: -0.0033 (p = 0.64) with 2 layers, +0.0009 (p = 0.82) with 3 layers.
- GATv2 does use the edges: rewiring them costs 0.024 (CI [+0.014, +0.033]). A plain mean already captures that information.
- Adding 40,030 unannotated windows to the graph lowers the uniform mean by 0.028 against the annotated-only 3x3 mean. More context was not better context.

## Embedding geometry

![Embedding geometry](figures/readme/06_latent_geometry.png)

Left: for 35 models, how often a score orders two models like their F1 does, with the sign and fit learned on the other 34 models. This analysis uses the earlier registry F1 values.

- Cluster scores do best: Fisher ratio 69.4%, Davies-Bouldin 67.4% (grouped with the neighbourhood measures in the figure), NC1 and Calinski-Harabasz 66.9%, silhouette 63.5%. Chance is 50%.
- Spectral scores are weaker (RankMe and the effective rank family, 59.5% at most). RankMe (37.2%) and LogME on train (33.6%) fall below chance because the sign learned on other models flips.
- These scores rank, they do not predict: the error gain over predicting the mean F1 is about 0.0004 at best, on a tier whose F1 range is about 0.04.
- Pairs share models, so they are not independent. The p-values come from a per-model Wilcoxon test with Benjamini-Hochberg correction inside each family, not from the pair count.

Right: keeping 100 principal components recovers 97.4% to 100.3% of the full-embedding F1 for 5 backbones.

## Data size

![Data curve](figures/readme/07_data_curve.png)

- LoRA is above the frozen probe at every fraction (+0.005 to +0.023). With 10% of the tiles LoRA reads 0.4725, about the frozen probe on all tiles (0.4712).
- Spatially blocked subsets are far worse than random ones: 0.099 against 0.426 at 1%, 0.434 against 0.480 at 50%. Blocked 1% subsets contain only 3 to 6 of the 8 main classes, so most of that gap is class coverage. I did not separate it from spatial leakage.

## Per class

| Class | Train tiles | Test tiles | Frozen DINOv3-H+ | Context fusion (fine-tuned) |
|---|---:|---:|---:|---:|
| ALDE | 5,285 | 1,884 | 0.958 | 0.962 |
| ARCA | 88 | 113 | 0.000 | 0.000 |
| BIRC | 12,845 | 4,937 | 0.837 | 0.867 |
| DRYI | 390 | 177 | 0.000 | 0.000 |
| LICH | 5,334 | 1,615 | 0.635 | 0.663 |
| MOSS | 5,089 | 1,935 | 0.520 | 0.600 |
| PETF | 1,990 | 1,037 | 0.506 | 0.572 |
| RUBC | 330 | 31 | 0.000 | 0.000 |
| SEDG | 5,250 | 2,134 | 0.601 | 0.694 |
| TUSS | 8,306 | 3,189 | 0.742 | 0.769 |
| WILL | 4,374 | 546 | 0.486 | 0.460 |

Three classes score exactly zero for these two models (all 3 seeds for the fine-tuned one, a single run for the frozen one). Over all 230 baseline probes the best values are 0.082 (ARCA), 0.026 (DRYI) and 0.146 (RUBC). They are the rarest classes, which pulls macro F1 down for every model, so compare models rather than reading the absolute level. A twelfth class, RHOL, is absent from val and test and is dropped.

### The same ranking on the 8 reliable classes

![8-class view](figures/readme/08_classes_8.png)

The figure shows the 42 configurations and, as red diamonds, the context, decoding and aggregation arms (each from its own run). The rank-change panel on the right covers the 42 configurations only.

ARCA, DRYI and RUBC score near zero for every model, so they lower the macro average without separating models. The 8-class view drops them (321 of 17,598 test tiles) and reports the macro F1 over the remaining 8, with the same 11-class classifier: their tiles stay in the test split, so a kept class they are predicted as counts as a false positive, and a kept tile predicted as one of the three counts as a false negative. The three classes are simply not scored. This is the convention of the `f1_macro_8cls_test` field that each training run writes. It is not a refit on 8 classes. Refits on the 8 classes exist from an earlier campaign (24 entries), but they follow a different protocol and are not used here.

- The level rises by about 0.18 (median +0.179): the 42 configurations span 0.3547 to 0.5079 in 11 classes and 0.4805 to 0.6984 in 8 classes.
- The order barely moves: Spearman 0.991, Kendall 0.965. 26 models keep their rank, 10 move one place, 4 move two, and 2 move three or more.
- The two movers are DINOv3-B with full fine-tuning (9 places down) and DINOv3-B with attention only (3 places down). The difference between the two rankings comes entirely from the three dropped classes, which weigh 3/11 of the 11-class average. DINOv3-B full has the highest ARCA score of the 42 (0.039), worth about +0.0035 on its own. Nine models have a non-zero ARCA score, none above 0.040.
- The best recipe (3x3 mean + mean field) reads 0.5359 in 11 classes and 0.7369 in 8 classes.

Rows are sorted by 8-class F1, ascending; the last column gives the run each value comes from.

| Arm | 11 classes | 8 classes | Run |
|---|---:|---:|---|
| GATv2, edges shuffled | 0.4648 | 0.6391 | aggregation |
| Single tile | 0.4719 | 0.6489 | decoding |
| CNN on the 3x3 window | 0.4824 | 0.6575 | probe-arms |
| GATv2, 2 layers | 0.4891 | 0.6725 | aggregation |
| Uniform mean over the graph | 0.4924 | 0.6770 | aggregation |
| GATv2, 3 layers | 0.4933 | 0.6783 | aggregation |
| Attention over the 3x3 window | 0.5085 | 0.6957 | probe-arms |
| Attention, positions shuffled | 0.5115 | 0.6999 | probe-arms |
| 3x3 mean | 0.5207 | 0.7160 | decoding |
| Single tile + mean field | 0.5297 | 0.7284 | decoding |
| 3x3 mean + ICM | 0.5332 | 0.7331 | decoding |
| 3x3 mean + mean field | 0.5359 | 0.7369 | decoding |

These arms are plotted as red diamonds in the figure above.

## Protocol

| Item | Value |
|---|---|
| Data | Arctic-TVC, drone orthomosaics at about 2.2 mm/px, Trail Valley Creek, NWT. [doi:10.20383/103.01418](https://doi.org/10.20383/103.01418) |
| Split | By whole orthomosaic, no tile leakage: 15 train, 8 val, 9 test. 49,281 / 13,209 / 17,598 tiles |
| Classes | 11 (RHOL removed). Metric: macro F1 over the 11 classes |
| Probe | Multinomial logistic regression (lbfgs), C in {1e-4, 1e-3, 0.01, 0.1, 1, 10} chosen on val |
| Intervals | Paired hierarchical bootstrap, orthomosaic then tile, 10,000 replicates, Benjamini-Hochberg across the pairs of a pre-declared family |
| Determinism | seed 42, single-thread BLAS (the thread count alone shifts F1 by about 0.0015) |

## Ongoing work

I am extending this toward transferability and alignment of Earth-observation foundation models under spatial and sensor shift.

- A first pilot found that a distribution discrepancy score (MMD, optimal transport) is associated with transfer, and the association replicated in a second experiment. I could not show that it adds value over RankMe, so I do not claim it.
- Training with an alignment loss is on hold until a discrepancy score shows value beyond RankMe.
- First frozen scores for three EO-oriented models are below DINOv3 (0.4805): SigLIP 2 ViT-B/16 0.4475, SatlasPretrain Aerial SwinB 0.4373, C-RADIO v4-H 0.3764. These are single-seed and not in the ranking above. Satlas normalisation is assumed to be ImageNet, and C-RADIO is run at 224 px instead of its native 512.
- `leo-explora-ssl/` holds exploratory self-supervised pretraining on aerial tiles (ExPLoRA style). No result from it is reported here.

## Limits

- 9 test maps cap what can be shown. Intervals are wide, and the percentile intervals are not centred on the estimate (for example +0.0152 with CI [+0.006, +0.062]). Read them as "excludes zero or not", not as a precision. For some pairs the interval and the p-value disagree (fine-tuned ResNet-50 ImageNet against the best frozen model: CI [-0.061, -0.015], p = 0.094). I report the Benjamini-Hochberg decision defined in the protocol.
- Model scores in the ranking and adaptation sections come from the clean baseline (2026-10-08). The geometry section uses the earlier registry values, which differ by at most 0.0043. The context, smoothing and aggregation sections come from their own runs: the same frozen single-tile SimDINOv2-B/16 reads 0.4718 to 0.4719 across the three, and the 3x3 mean reads 0.5205 and 0.5207. Comparisons are paired within a run.
- Frozen models have one run each, so their seed SD is not defined.
- Context, smoothing and GATv2 were measured on one frozen backbone, SimDINOv2-B/16. The gain on fine-tuned or other backbones is not measured.
- Invalid ExPLoRA runs and old single-seed fine-tunes are excluded everywhere. `tests/test_repo_guards.py` forbids a few retired values from reappearing.

## Reproduce

A clone contains the code, the configs, the 11-class splits and the scalar tables behind every number below. It does not contain the raw orthomosaics (public, FRDR), the 224 px tiles, the extracted embeddings, or the model weights (some are gated).

From a clone, the figures and every published number are recomputed from the committed tables:

```bash
pip install -r requirements.txt
python scripts/make_readme_plots.py      # rebuilds the eight figures from results/
```

The full pipeline starts from raw rasters. It needs the Arctic-TVC download, a tiling step, and a GPU for fine-tuning:

```bash
# tiles, then the split CSVs (12 classes), then the 11-class remap
python scripts/tile_split.py
python scripts/generate_dataset_splits.py
python scripts/generate_splits_11cls.py --splits-dir out/dataset_splits --out-dir splits --force

# embeddings and the canonical probe
python extract.py --config configs/frozen_dinov3_lvd.yaml
python probe.py   --config configs/frozen_eval.yaml
python analyze.py --config configs/frozen_eval.yaml
```

Two things to know:

- The committed splits are canonical. They come from `splits/split_v3_assignment.json` (15 / 8 / 9 orthomosaics, verified against the CSVs). Both the 12-class and the 11-class CSVs are included, and the command above regenerates the 11-class files byte-identically from the 12-class ones. Rebuilding the 12-class files from raw tiles is not scripted end to end: `scripts/generate_dataset_splits.py` still reads the older `splits/split_assignment.json`.
- Fine-tuning needs GPUs and the gated weights. The fine-tuned scores are committed seed by seed in `results/baseline_clean_2026-10-08/per_seed.csv`, so no GPU is needed to check them.

## Where the numbers come from

| Section | Values | File |
|---|---|---|
| Ranking, adaptation, per class, 8-class view | per-seed and per-class F1 | `results/baseline_clean_2026-10-08/per_seed.csv`, `per_class_f1.csv` |
| Significance of the ranking | paired differences, CIs, q-values | `results/baseline_clean_2026-10-08/statistics/pairwise_effects.csv` |
| Context, smoothing, learned aggregation | F1, deltas, CIs | `results/tile_context_2026-11-01/*.json` |
| Embedding geometry | pair ordering, leave-one-model-out | `results/rapport_data/loo_triage.csv`, `dim_probe_summary.csv` |
| Data size | learning curves, blocked subsets | `results/datacurve_lora/*_agg.csv`, `lora_spatial_v2/results_spatial_summary.csv`, `spatial_datacurve/class_composition.csv` |
| Per-class table | tile counts, per-class F1 | `splits/*_11cls.csv`, `per_class_f1.csv` |
| Non-linear readouts | 0 of 33, 2 of 24 | `results/rapport_data/head_sweep_summary.md` |

Code map: `src/` package, the `extract.py` / `probe.py` / `train.py` / `analyze.py` entry points, `configs/` YAML, `scripts/` analysis and SLURM jobs, `research/gnn_grid_2026-10-31/code/` tile context, smoothing and GNN experiments, `results/` scalar outputs.

Arctic-TVC is public on FRDR. Code is MIT, figures and results are CC-BY-4.0.
