# Validation leave-one-model-out du triage par géométrie

Palier `registry.TIER_K` = 35 modèles ; 1190 paires prédictives. Signe et régression (sur rangs) ajustés sur n−1 modèles, évalués sur le modèle exclu. p = Wilcoxon par modèle (n=35), BH par famille.

| Métrique | Famille | In-sample | LOO | MAE rangs | MAE moyenne | ρ(pred,F1) | p_BH(fam) |
|---|---|---|---|---|---|---|---|
| Calinski-Harabasz | séparabilité | 69 % | 69.1 % | 0.0061 | 0.0061 | +0.439 | 2.06e-05 |
| NC1 (intra/inter) | séparabilité | 69 % | 68.7 % | 0.0061 | 0.0061 | +0.426 | 2.06e-05 |
| Ratio de Fisher | séparabilité | 69 % | 68.6 % | 0.0059 | 0.0061 | +0.439 | 2.06e-05 |
| Davies-Bouldin | voisinage | 67 % | 67.1 % | 0.0060 | 0.0061 | +0.359 | 6.41e-05 |
| Stable rank | spectrale | 36 % | 64.0 % | 0.0061 | 0.0061 | +0.255 | 1.63e-03 |
| Participation ratio | spectrale | 36 % | 63.5 % | 0.0061 | 0.0061 | +0.271 | 1.63e-03 |
| Silhouette | séparabilité | 62 % | 61.8 % | 0.0061 | 0.0061 | +0.243 | 4.69e-05 |
| Pureté k-NN | voisinage | 61 % | 61.0 % | 0.0065 | 0.0061 | +0.236 | 3.35e-02 |
| Dim. intrinsèque | voisinage | 58 % | 58.5 % | 0.0069 | 0.0061 | +0.106 | 3.32e-02 |
| RankMe (σ) | spectrale | 57 % | 56.6 % | 0.0065 | 0.0061 | +0.116 | 1.91e-01 |
| Rang effectif (σ²) | spectrale | 44 % | 56.3 % | 0.0062 | 0.0061 | -0.505 | 8.42e-02 |
| Anisotropie | voisinage | 52 % | 52.4 % | 0.0063 | 0.0061 | -0.875 | 7.93e-01 |
| LogME (train) | transférabilité | 49 % | 44.4 % | 0.0062 | 0.0061 | -0.389 | 1.57e-01 |
| LogME (test) | transférabilité | 50 % | 42.7 % | 0.0063 | 0.0061 | -0.313 | 4.63e-02 |

## Lecture

- **Séparabilité (Fisher, DBI, CHI, NC1, silhouette) : la validation LOO passe.** 62 à 69 % d'ordonnancement hors échantillon, p_BH ≤ 4.7e-05 ; la régression sur rangs bat la moyenne pour 1/4 métriques.
- **Elle classe, elle ne prédit pas finement le F1** : gain de MAE ≤ 0.0001 sur un palier dont l'étendue de F1 vaut 0.039.
- **Spectre et transférabilité** : Stable rank 64.0 %, Participation ratio 63.5 %, RankMe (σ) 56.6 %, Rang effectif (σ²) 56.3 %. Sous le hasard : LogME (train) 44.4 %, LogME (test) 42.7 %.

Généré par `scripts/rapport/loo_triage.py`.
