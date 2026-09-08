# Probe MLP vs probe linéaire — `results/rapport_data/`

Sur les 20 modèles du tableau canonique, le delta brut MLP − probe lbfgs est négatif partout (médiane -0.0041). **Ce signe est un artefact d'optimiseur, pas une propriété des représentations.** Décomposé par un témoin apparié :

| Composante | Définition | Médiane | Étendue |
|---|---|---|---|
| `delta_protocole` | linéaire(AdamW) − linéaire(lbfgs) | -0.0061 | [-0.0129, -0.0002] |
| `delta_nonlinearite` | MLP(ReLU) − linéaire(AdamW) | +0.0032 | [-0.0067, +0.0082] |
| `delta` (brut, trompeur) | MLP(ReLU) − linéaire(lbfgs) | -0.0041 | [-0.0088, -0.0016] |

Un readout **strictement linéaire** entraîné par la même boucle (AdamW, early stopping, patience 15) perd 0.0061 de F1 face au même modèle linéaire résolu à l'optimum par lbfgs. Comparer directement MLP(AdamW) et probe(lbfgs) fait donc varier **deux** choses à la fois. À protocole égal, la ReLU apporte +0.0032 en médiane (14/20 modèles positifs).

## Effet de la non-linéarité, par modèle

| Modèle | MLP | linéaire (AdamW) | linéaire (lbfgs) | Δ non-lin. | Δ protocole |
|---|---|---|---|---|---|
| `simdinov2_vitb16_full` | 0.4766 | 0.4683 | 0.4782 | +0.0082 | -0.0099 |
| `resnet50_fulft_sota` | 0.4547 | 0.4475 | 0.4572 | +0.0072 | -0.0097 |
| `vitb16_scratch_old` | 0.3503 | 0.3439 | 0.3548 | +0.0064 | -0.0109 |
| `simdinov2_vitb16` | 0.4681 | 0.4639 | 0.4723 | +0.0042 | -0.0084 |
| `dinov3_vitb16_lvd_full` | 0.4765 | 0.4724 | 0.4788 | +0.0041 | -0.0064 |
| `simdinov2_vitl16` | 0.4738 | 0.4696 | 0.4760 | +0.0041 | -0.0064 |
| `dinov3_vitl16_sat` | 0.4532 | 0.4491 | 0.4620 | +0.0041 | -0.0129 |
| `dinov3_vitb16_lvd_explora` | 0.4787 | 0.4748 | 0.4816 | +0.0039 | -0.0068 |
| `vitb16_explora_old` | 0.4659 | 0.4624 | 0.4705 | +0.0034 | -0.0081 |
| `dinov3_vitb16_lvd_mhsa` | 0.4769 | 0.4736 | 0.4790 | +0.0033 | -0.0053 |
| `dinov3_vitb16_lvd` | 0.4687 | 0.4656 | 0.4712 | +0.0031 | -0.0057 |
| `dinov3_vitb16_lvd_lora_r8` | 0.4796 | 0.4766 | 0.4820 | +0.0030 | -0.0054 |
| `resnet50_imagenet` | 0.4007 | 0.3995 | 0.4076 | +0.0012 | -0.0081 |
| `vitb16_full_old` | 0.4707 | 0.4703 | 0.4748 | +0.0004 | -0.0045 |
| `vitb16_mhsa_old` | 0.4641 | 0.4641 | 0.4689 | -0.0001 | -0.0047 |
| `dinov3_vitl16_lvd` | 0.4717 | 0.4734 | 0.4792 | -0.0017 | -0.0058 |
| `simdinov2_vitb16_mhsa` | 0.4664 | 0.4685 | 0.4691 | -0.0021 | -0.0005 |
| `scalemae_vitl16` | 0.4394 | 0.4453 | 0.4480 | -0.0059 | -0.0028 |
| `satmae_vitl16` | 0.4008 | 0.4069 | 0.4091 | -0.0061 | -0.0022 |
| `vitb16_imagenet` | 0.4431 | 0.4497 | 0.4500 | -0.0067 | -0.0002 |

**0/20** modèles dépassent le seuil d'interprétabilité de 0,01 (AGENTS.md §4.4, σ inter-seed ≈ 0,008). Le gain non linéaire est donc **réel dans son signe mais trop faible pour être cité modèle par modèle** — seule sa constance sur 20 modèles est informative.

## Répartition par classe (Δ non-linéarité, moyenne sur les modèles)

| Classe | Δ moyen |
|---|---|
| WILL | +0.0097 |
| BIRC | +0.0075 |
| ALDE | +0.0075 |
| PETF | +0.0032 |
| TUSS | +0.0032 |
| SEDG | -0.0035 |
| LICH | -0.0037 |
| MOSS | -0.0039 |

Classes faibles (MOSS, WILL, PETF) : **+0.0030** ; les 5 autres classes vivantes : **+0.0022**. ARCA/DRYI/RUBC restent à 0 dans les trois readouts (`CLASSES_DEAD`).

## Corrélations Δ non-linéarité ↔ géométrie (population `all`)

Aucune métrique géométrique n'a d'IC95 bootstrap excluant 0. Les corrélations fortes de `correlations_mlp_delta_geometry.csv` (anisotropie, silhouette…) portent sur le delta **confondu** et reflètent surtout le coût de l'optimiseur — ne pas les citer comme un lien géométrie ↔ non-linéarité. —

## Contrôles

| Contrôle | Statut | Détail |
|---|---|---|
| labels train permutés — `dinov3_vitb16_lvd` | ✅ | test F1 = 0.0922 (hasard stratifié 0.0872), acc train = 0.370 |
| labels train permutés — `simdinov2_vitb16_full` | ✅ | test F1 = 0.0901 (hasard stratifié 0.0872), acc train = 0.419 |
| témoin linéaire-AdamW (sans ReLU) | ✅ | médiane -0.0061 vs lbfgs — isole le coût de l'optimiseur |

**Un readout non linéaire peu profond extrait un peu plus que le probe linéaire (≈ +0,004 de F1-macro), mais l'effet est sous le seuil de bruit inter-seed : il ne remet pas en cause le classement du benchmark.**

_Généré par `scripts/rapport/mlp_probe.py --decompose`, 2026-07-26 15:46._
