# Sonde tronquée par PCA — synthèse multi-modèles

F1-macro test (11 classes) par nombre de composantes principales ; `k_98`/`k_99` = plus petit `k` atteignant 98 %/99 % du F1 plein. Sonde canonique (lbfgs, grille C réduite) ; mono-thread.

| Modèle | dim | k=10 | k=20 | k=50 | k=100 | k=200 | k=400 | plein | % à k=100 | k_98 | k_99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov3_vitl16_lvd | 1024 | 0.403 | 0.447 | 0.464 | 0.470 | 0.475 | 0.479 | 0.479 | 98.2 | 100 | 200 |
| simdinov2_vitb16 | 768 | 0.447 | 0.455 | 0.464 | 0.473 | 0.467 | 0.470 | 0.472 | 100.3 | 50 | 100 |
| dinov3_vitb16_lvd | 768 | 0.408 | 0.446 | 0.463 | 0.467 | 0.467 | 0.469 | 0.471 | 99.1 | 50 | 100 |
| dinov3_vitl16_sat | 1024 | 0.335 | 0.389 | 0.439 | 0.450 | 0.457 | 0.460 | 0.462 | 97.4 | 200 | 400 |
| vitb16_imagenet | 768 | 0.364 | 0.395 | 0.430 | 0.440 | 0.451 | 0.452 | 0.450 | 97.8 | 200 | 200 |

Généré par `scripts/rapport/dim_probe_summary.py`.
