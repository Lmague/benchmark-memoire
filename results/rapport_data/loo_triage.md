# Validation leave-one-model-out du triage par géométrie

Palier `registry.TIER_K` = 35 modèles ; 1184 paires prédictives. Signe et régression (sur rangs) ajustés sur n−1 modèles, évalués sur le modèle exclu. p = Wilcoxon par modèle (n=35), BH par famille.

| Métrique | Famille | In-sample | LOO | MAE rangs | MAE moyenne | ρ(pred,F1) | p_BH(fam) |
|---|---|---|---|---|---|---|---|
| Ratio de Fisher | séparabilité | 69 % | 69.4 % | 0.0048 | 0.0052 | +0.415 | 1.32e-04 |
| Davies-Bouldin | voisinage | 67 % | 67.4 % | 0.0048 | 0.0052 | +0.334 | 1.09e-04 |
| Calinski-Harabasz | séparabilité | 67 % | 66.9 % | 0.0049 | 0.0052 | +0.356 | 1.32e-04 |
| NC1 (intra/inter) | séparabilité | 67 % | 66.9 % | 0.0049 | 0.0052 | +0.355 | 1.32e-04 |
| Silhouette | séparabilité | 64 % | 63.5 % | 0.0052 | 0.0052 | +0.284 | 1.32e-04 |
| Rang effectif (σ²) | spectrale | 41 % | 59.5 % | 0.0052 | 0.0052 | +0.049 | 1.51e-02 |
| Participation ratio | spectrale | 41 % | 59.1 % | 0.0053 | 0.0052 | +0.055 | 1.51e-02 |
| Dim. intrinsèque | voisinage | 58 % | 58.1 % | 0.0057 | 0.0052 | +0.087 | 4.31e-02 |
| Stable rank | spectrale | 42 % | 57.8 % | 0.0054 | 0.0052 | +0.001 | 2.66e-02 |
| Pureté k-NN | voisinage | 58 % | 57.8 % | 0.0055 | 0.0052 | +0.187 | 8.39e-02 |
| Anisotropie | voisinage | 57 % | 56.9 % | 0.0054 | 0.0052 | +0.126 | 7.44e-02 |
| LogME (test) | transférabilité | 56 % | 55.7 % | 0.0053 | 0.0052 | -0.211 | 6.17e-02 |
| RankMe (σ) | spectrale | 50 % | 37.2 % | 0.0055 | 0.0052 | -0.365 | 1.13e-02 |
| LogME (train) | transférabilité | 50 % | 33.6 % | 0.0054 | 0.0052 | -0.474 | 2.62e-06 |

## Lecture

- **Séparabilité (Fisher, DBI, CHI, NC1, silhouette) : la validation LOO passe.** 64–69 % d'ordonnancement hors échantillon, signe stable, p_BH ≤ 1,3×10⁻⁴ ; la régression sur rangs bat la moyenne pour 4/5 métriques.
- **Elle classe, elle ne prédit pas finement le F1** : gain de MAE ≤ 0,0004 sur un palier dont l'étendue de F1 vaut ≈ 0,04.
- **Spectre : inexploitable dans le sens naturel** (58–59 % seulement avec le signe appris, ≤ 42 % dans le sens pré-enregistré) ; RankMe et LogME train ont un signe instable (pires que le hasard).

Généré par `scripts/rapport/loo_triage.py`.
