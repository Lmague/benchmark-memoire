# Arctic tundra vegetation mapping with vision foundation models

This repository holds the code and results of my Master's project at UQAM. The
question was practical: to map Arctic tundra vegetation from drone imagery, do you
need to fine-tune a vision model on your own data, or is a frozen pretrained
backbone good enough?

The short answer is that fine-tuning helps a little, but the thing that really
moves the ceiling is spatial context, meaning the neighbouring area around each
tile.

## Setup

Arctic-TVC: 38 drone orthomosaics at about 2.2 mm per pixel, 35 annotated,
collected at Trail Valley Creek in the Northwest Territories. Each whole
orthomosaic goes to train, validation or test, so no tile from a test orthomosaic
can appear in training.

I compare 42 configurations: 11 frozen backbones and 31 fine-tuned ones with three
seeds each. Everything uses the same linear probe, and fine-tuned models are probed
again after training, so the number describes the representation rather than the
head used during training.

Results are macro F1 over the 11 classes present in the test split. The taxonomy
has 12, but RHOL is absent from the test set, and three classes (ARCA, DRYI, RUBC)
score near zero for every model, so the absolute F1 is lower than a cleaner
benchmark would give.

## Ranking

![All 42 configurations, same linear probe on the same test split](figures/fig_readme_ranking.png)

The ceiling comes from adding the neighbourhood: 0.5080 for context fusion, 0.5057
for the same idea on SimDINOv2-B. Among single-tile models, context distillation
reaches 0.4870 and DINOv3-L with LoRA reaches 0.4842, both slightly above the best
frozen model, DINOv3-H+/16 at 0.4805.

## What helped

### Spatial context

Fusing a tile with its 512 px neighbourhood adds 0.025 to 0.034. It is positive on
8 of the 9 test orthomosaics. A frozen SimDINOv2-B with a fused readout reaches
0.5059 without a single gradient step.

The controls matter more than the gain. Context alone is no better than the tile
alone (0.4781 against 0.4778), and a permuted context does worse than the tile
alone (0.4745). What helps is the aligned neighbourhood, not the extra pixels. On
five frozen backbones, 512 px beats 1024 px, which beats 2048 px.

![Spatial context, with and without the neighbourhood](figures/fig_readme_context.png)

### Model size

At the same architecture, LoRA r=8 gains 0.0114 over its own frozen DINOv3-B. But
a larger frozen model still wins: frozen ViT-L/16 sits 0.004 above frozen ViT-B/16
on 8 of the 9 sites, and full fine-tuning of the smaller model ends just below it
(0.4786 against 0.4792). At this scale, size buys what adaptation does not.

## What did not help

- **More LoRA.** Rank, scaling, block placement and Q+K+V targets all land within
  0.0067, and higher scaling made things worse. Updating only the normalization
  layers plus the head, 47k parameters, lands within a thousandth of LoRA rank 8.
- **Other readouts.** Cosine k-NN was 0.031 to 0.104 below the linear probe, and
  MLP-2, bilinear and FiLM heads were worse in 33 groups out of 33 (mean -0.0103).
- **Texture features and LogME.** 106 handcrafted texture variables changed nothing
  (-0.002 to +0.001); the embedding already predicts them with R-squared above
  0.88. LogME came out bimodal and could not rank models.
- **Satellite pretraining.** SatMAE 0.4091, Scale-MAE 0.4480, DINOv3-L satellite
  0.4620, against 0.4792 for the same architecture on natural images.
- **Training from scratch.** 0.3561, last of the ranking.

## Latent space

![Spectra, ranking power and probe dimension](figures/fig_readme_latent.png)

**Spectra.** The broken backbones give themselves away. Scale-MAE and the model
trained from scratch lose most of their energy in the first few dozen directions,
while the usable representations keep a long tail. That is the clearest geometry
signal, and it needs no labels.

**What predicts F1.** Inside the competitive tier, class separability is the only
family that ranks. Fisher ratio, Calinski-Harabasz, NC1 and Davies-Bouldin order 69
to 72 percent of model pairs like F1, and silhouette 60 percent. The spectral
family does not: effective rank 41 percent, participation ratio 33 percent,
intrinsic dimension 42 percent, and RankMe 55 percent, which is chance. Anisotropy
behaves the same way, informative on the full population and useless inside the
tier.

**Where geometry does work.** Within a single backbone, across its 12 layers,
RankMe tracks F1 with a Spearman rho of 0.89 and anisotropy with -0.92. The
geometry is a good map of where a model sits along its own depth, and a poor
instrument for comparing different models at the top.

**Dimension and redundancy.** Keeping 100 principal components reaches 99.1 percent
of the full probe's F1 on DINOv3-B; the remaining 668 dimensions add 0.004. And the
five separability measures agree at absolute rho above 0.98, so they carry one
piece of information, not five.

## Learning curves and sampling

![Data curves and spatially blocked sampling](figures/fig_readme_datacurve.png)

Fine-tuning a strong backbone costs a lot of data and returns little. LoRA needs
about 4,592 tiles just to match its own frozen backbone, and ends only 0.012 above
it. On the weaker ImageNet backbone the same regimes overtake their frozen baseline
at about 2,060 tiles and gain 0.025 to 0.028.

The random draw also flatters results. With a spatially blocked draw, performance
at 1 percent of the data falls from 0.426 to 0.099. Part of that is class coverage:
at 1 percent only 3 to 6 classes are present in training. Restricting the score to
the classes present raises it to 0.229, so the drop is first a coverage problem and
then a spatial one.

## What the numbers cannot support

The test set has nine orthomosaics. That is few, and it sets the limit of what can
be claimed.

**The marginal interval is not the right yardstick.** Cluster bootstrap over
orthomosaics widens the mean 95 percent interval on F1 from 0.0142 to 0.1788. But
both models are measured on the same sites, so the site difficulty cancels, and
paired per-site differences are much tighter. Nine sites is still few: the smallest
effect this design can detect is roughly 0.03 to 0.05.

**An earlier claim had to be withdrawn.** This README used to say that no pairwise
difference survived a cluster bootstrap over 42 models. That was misleading. The
bootstrap p-value has a resolution of 0.0002, while the correction over 595 pairs
needed 0.00008 at the top rank, so almost nothing could survive regardless of the
data. The honest reading is that the tier cannot be ranked at this sample size.

**The choice of C is unstable.** DINOv3-L LoRA came out at 0.4842 against 0.4799
published, entirely because of which C was kept per seed. Treat 0.005-level
claims as provisional.

**Three classes are dead.** ARCA, DRYI and RUBC are near zero for every model, and
RHOL is absent. Macro F1 is pulled down by classes no model can learn here.

## Running it

```bash
pip install -r requirements.txt

python extract.py --config configs/frozen_dinov3_lvd.yaml
python probe.py   --config configs/frozen_eval.yaml
python analyze.py --config configs/frozen_eval.yaml
```

Splits and embeddings are not included: regenerate them with `extract.py`. Two
rules matter. The probe forces single-threaded BLAS, because the thread count
changes the floating point reduction order and moves lbfgs; the effect is about
0.0015 of F1. And the regularization grid must be the same for every model, with
six points, C in {1e-4, 1e-3, 0.01, 0.1, 1, 10}. `tests/test_repo_guards.py`
checks both.

`scripts/make_readme_figures.py` regenerates the four figures above.

## Data and licence

Arctic-TVC is public on FRDR:
[doi:10.20383/103.01418](https://doi.org/10.20383/103.01418). Code is MIT, figures
and results are CC-BY-4.0.
