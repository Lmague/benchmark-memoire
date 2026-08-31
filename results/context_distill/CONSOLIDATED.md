## 1. F1 des runs contexte (ctx1024, LoRA r2a4, blocs 6-11, 3 seeds)
| Design | Teacher | seed0 | seed1 | seed2 | moy ± std | F1_8cls moy | BAcc moy |
| A | dinov3_vitl16_lvd | 0.4855 | 0.4874 | 0.4880 | 0.4870 ± 0.0011 | 0.6696 | 0.4913 |
| A | ema_self | 0.4854 | 0.4835 | 0.4819 | 0.4836 ± 0.0014 | 0.6650 | 0.4877 |
| B | dinov3_vitl16_lvd | 0.5098 | 0.5070 | 0.5071 | 0.5080 ± 0.0013 | 0.6985 | 0.5077 |

## 2. Matrice d'attribution (seed0, même test v3, même sonde canonique)
| Modèle | tile (768) | fused (1536) | Δ contexte | best_C tile | best_C fused |
| FROZEN DINOv3-B | 0.4716 | 0.4862 | +0.0145 | 0.001 | 0.001 |
| LoRA r2 (v3train) | 0.4875 | 0.4946 | +0.0072 | 0.0001 | 0.001 |
| LoRA r8a16 (spatial) | 0.4887 | 0.4945 | +0.0058 | 0.001 | 0.001 |
| R1 dA_tL (distill A) | 0.4856 | 0.4952 | +0.0097 | 0.001 | 0.0001 |
| R2 dB_tL (distill B) | 0.4779 | 0.5098* | — | 0.001 | — |
\* R2 fused = 0.5098 = moyenne 3 seeds des metrics.json (matrice = tile seulement pour R2).

## 3. Géométrie de l'espace latent (test v3, subsample 20k, seed 42)
| Modèle | repr | dim | RankMe | RankMe/D | Aniso | α-ReQ | NESum |
| FROZEN | tile | 768.0 | 357.6 | 0.4656 | +0.5873 | 1.6550 | 5.85 |
| dA_tEMA_seed0 | tile | 768 | 340.2 | 0.4430 | +0.3312 | 1.7672 | 4.44 |
| dA_tEMA_seed1 | tile | 768 | 359.5 | 0.4681 | +0.3300 | 1.7077 | 4.74 |
| dA_tEMA_seed2 | tile | 768 | 328.4 | 0.4276 | +0.3856 | 1.7727 | 4.02 |
| dA_tL_seed0 | tile | 768 | 350.7 | 0.4566 | +0.3327 | 1.7419 | 4.63 |
| dA_tL_seed1 | tile | 768 | 366.1 | 0.4766 | +0.3390 | 1.7022 | 5.09 |
| dA_tL_seed2 | tile | 768 | 350.2 | 0.4560 | +0.3291 | 1.7355 | 4.46 |
| dB_tL_seed0 | fused | 1536 | 723.5 | 0.4710 | +0.3439 | 1.7640 | 4.90 |
| dB_tL_seed0 | tile_only | 768 | 345.2 | 0.4495 | +0.4225 | 1.7721 | 4.63 |
| dB_tL_seed1 | fused | 1536 | 665.8 | 0.4335 | +0.4028 | 1.8139 | 3.75 |
| dB_tL_seed1 | tile_only | 768 | 330.7 | 0.4306 | +0.4626 | 1.7691 | 3.91 |
| dB_tL_seed2 | fused | 1536 | 715.4 | 0.4657 | +0.3898 | 1.7360 | 4.07 |
| dB_tL_seed2 | tile_only | 768 | 343.9 | 0.4478 | +0.4612 | 1.7300 | 4.30 |