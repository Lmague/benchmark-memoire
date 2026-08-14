# Courbe de données spatiale v2 — rapport des résultats

Date : 2026-08-14 · Runs : LoRA DINOv3-B r=8 α=16, 21 runs sur Narval
(7 fractions × 3 seeds, demi-A100, array SLURM `spatial_datacurve/slurm_datacurve_spatial_v2.sh`).

## Statut

- **20 runs sur 21 terminés.** `frac100_seed2` a échoué (dossier vide, pas de
  metrics.json) : le job a atteint le mur des 6 h après les 2 premiers seeds
  (prédit — frac100 ≈ 2 h/seed × 3 > 6 h).
- Les 2 seeds terminés à 100 % donnent 0,485 ± 0,005, cohérent avec le canonique
  0,4835 ± 0,0011 (même recette, même train).
- **Reprise** (quand voulu) : resoumettre le script — `--skip-if-done` reprend
  uniquement le seed manquant (les 20 autres sont protégés par la sentinelle done) :
  ```bash
  sbatch spatial_datacurve/slurm_datacurve_spatial_v2.sh
  ```

## Résultats (F1-macro test, moyenne inter-seed ± écart-type)

| Cible | Frac. réelle | Classes au train (11 cls) | F1 complet | F1 8 cls | F1 train-pres |
|---|---|---|---|---|---|
| 1 % | 1,0 % | 3–6 | 0,099 ± 0,052 | 0,136 ± 0,072 | 0,229 ± 0,091 |
| 5 % | 5,0 % | 6–7 | 0,170 ± 0,061 | 0,232 ± 0,086 | 0,290 ± 0,081 |
| 10 % | 10,0 % | 6–9 | 0,197 ± 0,164 | 0,270 ± 0,226 | 0,283 ± 0,171 |
| 25 % | 25,0–25,4 % | 11 | 0,372 ± 0,066 | 0,512 ± 0,091 | 0,372 ± 0,066 |
| 50 % | 49,6–50,8 % | 11 | 0,434 ± 0,022 | 0,596 ± 0,031 | 0,434 ± 0,022 |
| 75 % | 74,2–75,0 % | 11 | 0,462 ± 0,013 | 0,635 ± 0,018 | 0,462 ± 0,013 |
| 100 % | 100 % | 11 | 0,485 ± 0,005 (2 seeds) | 0,667 ± 0,007 | 0,485 ± 0,005 |

Références : courbe aléatoire stratifiée (r8a16) : 0,426 (1 %) → 0,473 (10 %) →
0,483 (100 %) ; DINOv3-B gelé 100 % = 0,4712.

## Lectures

1. **Le tirage aléatoire de tuiles surestime la performance** : Δ = +0,33 pt à
   1 %, +0,28 à 10 %, +0,10 à 25 %, ~0,02 à 75 %, nul à 100 %. C'est l'ampleur
   du confondant d'autocorrélation spatiale (chevauchement 50 % + classes
   présentes partout en stratifié).
2. **Plateau décalé** : la courbe spatiale n'atteint son plateau qu'à partir de
   50–75 % du volume (plusieurs orthomosaïques, toutes classes vues), contre
   ~5 % pour la courbe aléatoire.
3. **La variance aux petits volumes est dominée par le SITE** : écart-type
   inter-seed jusqu'à 0,16 (10 %), vs ±0,008 en aléatoire. Chaque seed tire un
   bloc d'un site différent → la performance dépend de *où* on collecte.
4. **Décomposition couverture vs volume** (colonne train-pres) : même en ne
   comptant que les classes présentes au train, le F1 à 1–10 % (0,23–0,29)
   reste sous la courbe aléatoire (0,43–0,47) : l'homogénéité spatiale du petit
   volume coûte aussi sur les classes vues. La couverture explique la moitié de
   la chute, l'homogénéité l'autre moitié.
5. **Convergence** : à 100 %, spatiale = aléatoire = canonique (mêmes données) —
   le design est sain.

## Livrables

- `lora_spatial_v2/results_spatial_summary.csv` — 20 runs agrégés (cible,
  fraction spatiale réelle, seed, tuiles 11 cls, F1 pres, F1 8 cls, F1
  train-pres, best_epoch, best_C).
- `rapport_bouguessa/figs/fig_spatial_{vs_random_11cls,8cls,train_pres}.png` —
  3 figures (générées par `rapport_bouguessa/figs/make_spatial_results_figs.py`).
- `rapport_bouguessa/tables/t_spatial_results.tex` — table des résultats.
- `rapport_bouguessa/datacurves.pdf` — section « Courbe de données spatiale :
  design et résultats » (12 pages au total).
