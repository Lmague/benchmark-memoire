# results/ — Carte des résultats actifs

Dernière mise à jour : 2026-09-13 (audit de cohérence — voir `docs/audit/corrections_2026-09-13.md`).

Ce dossier contient UNIQUEMENT les résultats qui font autorité pour le manuscrit
courant. Tout le reste est dans `_anciennes_experiences/` (voir
`_anciennes_experiences/ARCHIVE_INDEX.md`).

---

## Structure

```
results/
├── README.md                          ← ce fichier
├── T2a_kmeans_control.json            ← contrôle k-means (2026-07-14)
├── T2c_headline_pair_p_CANONICAL.json ← paire phare canonique (2026-07-14)
├── bootstrap_ci_2026-07-09.json       ← CI bootstrap final (2026-07-09)
├── correlations.json                  ← corrélations 12 modèles (2026-07-07)
├── geometry_extended_12models.json    ← géométrie latente 12 modèles (2026-07-07)
├── significance_matrix_all12.json     ← matrice de significativité (2026-07-07)
├── significance_matrix_all12.png      ← heatmap significativité
├── all_scores_consolidated.json       ← scores F1 consolidés (2026-07-07)
├── f1_learnable_12models.csv          ← F1 3 fine-tunés (2026-06-14)
├── correlations_n9_vs_n12.csv         ← comparaison corrélations n9/n12 (2026-06-14)
├── tiles_per_class_per_split.json     ← distribution tuiles (2026-06-15)
├── relance2/                          ← campagne de référence (2026-07-07)
├── final_2026-07/                     ← résultats finaux (2026-07-08 à 2026-07-10)
├── datacurve/                         ← courbes d'apprentissage (2026-06-20 à 2026-07-11)
├── figures_paper/                     ← figures du manuscrit (2026-07-07)
│
├── geometry_extended.json             ← [BLOQUÉ] scripts/compute_multiple_correction.py:44
├── literature_experiments.json/.csv   ← [BLOQUÉ] scripts/compute_all_correlations.py
├── sample_efficiency.json             ← [BLOQUÉ] scripts/sample_efficiency.py:121
├── significance_matrix.json           ← [BLOQUÉ] scripts/make_support_figures.py:121
└── significance_matrix.png            ← [BLOQUÉ] compagnon du précédent
```

## Source canonique de chaque chiffre du manuscrit

| Chiffre | Source canonique |
|---|---|
| F1 macro 12 modèles canoniques (11 classes) | `all_models_canonical_merged.json` / `registry.CANONICAL_F1` (valeurs `relance2`, source du manuscrit) ; `without_rhol/probe_knn_cgrid.json` = reprobe diagnostique (écarts ≤ 0,0003 sur 5 modèles) |
| F1 macro 12 modèles canoniques (8 classes) | `8cls/probe_knn_cgrid.json` |
| Validation LOO du triage par géométrie | `rapport_data/loo_triage.json` (`scripts/rapport/loo_triage.py`) |
| Synthèse des campagnes « têtes » non linéaires | `rapport_data/head_sweep_summary.json` (`scripts/rapport/head_sweep_summary.py`) |
| Classement maître 42 modèles (11 classes) | `all_models_canonical_merged.json` + `rapport_bouguessa/tables/t_master.tex` |
| Palier compétitif + bootstrap apparié (35 groupes, 595 paires) | `significance_matrix_tier.json` (n=10 000, BH α=0,05, 182 rejets) |
| Bootstrap CI (F1, 11cls) | `bootstrap_ci_2026-07-09.json` |
| Paire phare (ViT-B-FT ≈ DINOv3-L-LVD) | `T2c_headline_pair_p_CANONICAL.json` — contient UNIQUEMENT cette paire |
| Contrôle (ViT-B/16 vs ViT-B-FT) | `significance_matrix_all12.json` — paire `dinov3_vitb16_lvd` vs `vitb16_fulft_arctic`, Δ=-0.0076, p=0.003 |
| Corrélations LogME/F1, silhouette/F1… (n=12) | `relance2/relance2_correlations_11cls_n12.json` — silhouette ρ=0.825, p=0.001 |
| Géométrie spectrale (α-ReQ, NESum, RankMe…) | `geometry_extended_12models.json` |
| Géométrie cluster (silhouette, DB, CH, NC1, NC2) | `final_2026-07/data/geometry_11cls.json` |
| Significativité (BH, α=0,05) | `significance_matrix_all12.json` (l'ancienne correction Holm-Bonferroni est dans `multiple_comparison.py`, non utilisée par le pipeline actuel) |
| Courbes d'apprentissage | `datacurve/` → `paper_arctic_fm_benchmark/fig2_data_curve_regimes.png` |
| Contrôle k-means | `T2a_kmeans_control.json` |

## Fichiers bloqués

Les 6 fichiers marqués `[BLOQUÉ]` sont référencés par des chemins hardcodés dans
`scripts/`. Ils restent en place pour ne pas casser les pipelines existants.

## Archives

Tout l'historique est dans `_anciennes_experiences/ARCHIVE_INDEX.md`.