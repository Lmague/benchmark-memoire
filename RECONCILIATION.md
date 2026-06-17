# RECONCILIATION.md — Sanity-check post-passe nocturne 2 (17 juin 2026)

Toutes les valeurs proviennent des JSON **canoniques recalculés** depuis les 
embeddings (lecture seule). Aucune valeur n'a été transcritée à la main : ce 
document est généré par `scripts/make_reconciliation.py`.


---

## 1. F1 canonique (f1_macro_pres test, best_C par val)

| Modèle               | F1 recalculé | F1 contrôle | Δ       | best_C recalc | best_C ctrl | verdict |
|----------------------|--------------|-------------|---------|---------------|-------------|---------|
| resnet50_imagenet    | 0.4081      | 0.4080      | +0.0001 | 0.001         | 0.001       | OK      |
| vitb16_imagenet      | 0.4500      | 0.4503      | -0.0003 | 0.01          | 0.01        | OK      |
| dinov3_vitb16_lvd    | 0.4715      | 0.4714      | +0.0001 | 0.001         | 0.001       | OK      |
| dinov3_vitl16_sat    | 0.4620      | 0.4619      | +0.0001 | 0.001         | 0.001       | OK      |
| dinov3_vitl16_lvd    | 0.4789      | 0.4789      | -0.0000 | 0.001         | 0.001       | OK      |
| simdinov2_vitb16     | 0.4716      | 0.4714      | +0.0002 | 0.001         | 0.001       | OK      |
| simdinov2_vitl16     | 0.4761      | 0.4762      | -0.0001 | 0.001         | 0.001       | OK      |
| satmae_vitl16        | 0.4093      | 0.4094      | -0.0001 | 0.01          | 0.01        | OK      |
| scalemae_vitl16      | 0.4485      | 0.4481      | +0.0004 | 0.01          | 0.01        | OK      |
| resnet50_arctic      | 0.4619      | 0.4620      | -0.0001 | 0.0001        | 0.0001      | OK      |
| vitb16_arctic        | 0.4758      | 0.4758      | -0.0000 | 0.001         | 0.001       | OK      |
| vitb16_fulft_arctic  | 0.4791      | 0.4791      | +0.0000 | 0.001         | 0.001       | OK      |

**Verdict** : 0/12 modèles hors tolérance ±0.001.


---

## Paire PHARE (`dinov3_vitl16_lvd:vitb16_fulft_arctic`)

| Quantité               | Recalculé          | Contrôle           | Δ          | verdict |
|------------------------|--------------------|--------------------|------------|---------|
| Δ observé (A - B)      | -0.0002            | -0.0002            | +0.0000     | OK  |
| P(gelé > FT)           | 0.483              | 0.483              | +0.000      | OK  |
| CI95 modèle A          | [0.4730, 0.4852] | [0.4730, 0.4852] | -- | OK  |
| CI95 modèle B          | [0.4728, 0.4847] | [0.4728, 0.4847] | -- | OK  |
| IC95 disjoints         | False              | attendu = False    | --         | OK  |


---

## Paire CONTRÔLÉE (best_C) (`dinov3_vitb16_lvd:vitb16_fulft_arctic`)

| Quantité               | Recalculé          | Contrôle           | Δ          | verdict |
|------------------------|--------------------|--------------------|------------|---------|
| Δ observé (A - B)      | -0.0076            | -0.0076            | -0.0000     | OK  |
| P(gelé > FT)           | 0.003              | 0.003              | +0.000      | OK  |
| CI95 modèle A          | [0.4657, 0.4773] | [0.4657, 0.4773] | -- | OK  |
| CI95 modèle B          | [0.4728, 0.4847] | [0.4728, 0.4847] | -- | OK  |
| IC95 disjoints         | False              | attendu = False    | --         | OK  |


---

## 3. Corrélations métrique ↔ F1 (n=12, F1 canonique)

| Métrique           | ρ recalculé | ρ contrôle | Δ       | verdict |
|--------------------|-------------|------------|---------|---------|
| logme              | +0.762      | +0.78      | -0.018  | OK      |
| nesum              | +0.559      | +0.55      | +0.009  | OK      |
| alpha_req          | -0.615      | -0.65      | +0.035  | OK      |
| rankme_normalized  | +0.497      | +0.54      | -0.043  | OK      |
| rankme             | +0.266      | +0.32      | -0.054  | WARN    |
| anisotropy         | -0.538      | -0.60      | +0.062  | WARN    |

**Verdict** : 2/6 métriques hors tolérance ±0.05.


---

## 4. Corrélations top-K (paliers compétitifs)

| Métrique           | top-8 ρ  | top-6 ρ  |
|--------------------|----------|----------|
| logme              | +0.214    | +0.429    |
| rankme             | -0.262    | +0.086    |
| rankme_normalized  | -0.095    | -0.486    |
| alpha_req          | +0.214    | -0.029    |
| nesum              | -0.024    | -0.314    |
| anisotropy         | +0.143    | +0.143    |


---

## Légende


- `OK ` (vert) : dans la tolérance (±0.001 F1, ±0.05 ρ, ±0.005 p-value, ±0.005 CI).

- `!! ` (rouge) : hors tolérance — à investiguer ou documenter.

- `--` : non applicable ou non comparable.



## Sources canoniques


- F1 : `results/with_rhol/probe_knn_cgrid.json` (probe.py, grille étendue C∈{1e-4..10}, best_C par val).

- Paires : `results/significance_matrix_all12.json` (significance_matrix.py, bootstrap apparié n=1000, seed=42).

- Corrélations : `results/transfer/correlations_with_f1.json` + `results/transfer/logme_vs_f1.json` (task_a + task_b).

- Paliers compétitifs : `results/correlations.json` (compute_correlations.py).
