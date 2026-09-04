# spatial_datacurve/ — Courbe de données SPATIALE v2

Volume de train contrôlé par **unités spatiales** (orthomosaïques entières ou
blocs spatiaux contigus), jamais par tuiles tirées aléatoirement, pour éliminer
le confondant d'autocorrélation spatiale (tilerization 224 px, stride 112 px =
chevauchement 50 % entre tuiles voisines).

Générateur : `make_spatial_datacurve.py` (autonome, lit `splits/` canonique).
Plot : `make_sanity_plot.py`.

## Design

| Cible | Mécanisme | Détail |
|---|---|---|
| 1, 5, 10 % | bloc spatial contigu | `round(cible × 49433)` tuiles, intervalle d'indices `[offset, offset+k)` dans l'ordre row-major d'UNE ortho (cf. `scripts/tilerization.py`) |
| 25, 50, 75, 100 % | orthos entières | closest-sum : sous-ensemble des 15 orthos de train dont la somme de tuiles est la plus proche de la cible (2^15 énuméré) |

Seeds : seed 0 = solution déterministe (plus proche / plus grosse ortho, offset 0) ;
seeds 1-2 = tirages seedés parmi les combinaisons à ±1 % du volume total (orthos)
ou ortho+offset aléatoires (blocs), en excluant les tirages déjà pris.

Val/test : copie EXACTE du split canonique v3 (`splits/val.csv`, `splits/test.csv`),
aucune orthomosaïque partagée avec le train (vérifié par assertion + script).

## Contenu

```
spatial_datacurve/
├── make_spatial_datacurve.py
├── make_sanity_plot.py
├── manifest.json              ← manifest global (protocole + niveaux)
├── summary_table.csv / .md    ← table résumé (cible, fraction réelle, seed, unités, tuiles)
├── sanity_check.png           ← plot (a) fraction réelle, (b) décomposition, (c) classes
└── splits/fracXXX_seed{S}/    ← 21 splits (7 fractions × 3 seeds)
    ├── train.csv              ← tuiles sélectionnées (filepath,label) — le manifest du train
    ├── val.csv / test.csv     ← copies du canonique v3
    └── manifest.json          ← tile IDs EXACTS + sélection + composition 12/11/8 classes
```

## Notes scientifiques

- **Fractions réelles** (volume canonique = 49 433 tuiles, 15 orthos) :
  1 % (494 tuiles), 5 % (2 472), 10 % (4 943), 25 % (12 358), 50 % (24 716),
  75 % (37 075), 100 % (49 433). Les seeds 0 des niveaux 25/50/75 % atteignent
  la cible exacte ; les seeds 1-2 varient de ±1 pt.
- **Plancher ortho** : la plus grosse ortho = 9 029 tuiles ≈ 18 % → les cibles
  < 25 % ne peuvent PAS être des orthos entières (d'où les blocs).
- **Blocs contigus** : l'ordre `tile_XXXXX` est l'ordre de scan row-major de la
  tilerization (vérifié dans `scripts/tilerization.py:117-163`) ; un intervalle
  d'indices = bande spatiale contiguë. Les tuiles vides sautées créent des trous
  d'indices ABSOLUS, rapportés dans `max_gap_indices_absolus`.
- **Composition de classes** : à 1 %, un bloc contigu ne couvre que 3-6 des
  8 classes (dominance BIRC dans `birch24`) ; à 25 % et au-delà, les 8 classes
  sont présentes. C'est la conséquence voulue de l'échantillonnage spatial :
  à petit volume, le train est spatialement homogène et déséquilibré. La
  composition complète est dans chaque `splits/fracXXX_seed{S}/manifest.json`
  (schémas 12, 11 et 8 classes) et agrégée dans `class_composition.csv`.
- **Schéma de classes** : les CSV gardent les labels 12-classes canoniques ;
  le remapping 11/8-classes se fait au probe (cf. `scripts/datacurve_one_run.py`).

## Interprétation du déséquilibre de classes (décision : conserver tel quel)

La sélection ne contraint JAMAIS la composition en classes (c'est un choix
assumé, validé 2026-08-13) : dès qu'on contrôle le volume par unité spatiale,
la distribution des classes est celle du terrain sélectionné. Forcer la présence
de toutes les classes réintroduirait un biais de sélection vers l'équilibre et
effacerait l'effet qu'on mesure.

Faits à connaître (détail par split dans `class_composition.csv`) :
- à 1 %, le seed 0 (birch24, offset 0) est quasi monospécifique (BIRC 94 %,
  3 classes présentes) ; les seeds 1-2 couvrent 5-6 classes. Le cas dégénéré
  n'est donc pas universel, mais il EXISTE — c'est un tirage spatial réel.
- à 5 % et au-delà, les 8 classes sont présentes (7/8 pour un seed de frac001) ;
  les proportions restent biaisées jusqu'à ~50 % (ex. BIRC 71 % à 25 % vs 26 %
  à 100 %) puis convergent vers la composition du train complet.

Comment comparer quand même les fractions (protocole d'évaluation recommandé) :
1. Le test est TOUJOURS le même (9 orthos du split v3, toutes les classes) :
   chaque run produit un F1 macro sur le même test → comparable entre fractions.
2. Calculer DEUX métriques par run (convention « pres » déjà utilisée dans le
   dépôt pour RHOL, cf. `f1_macro_pres` dans `datacurve_one_run.py`) :
   - F1 macro strict sur les 8 classes du test → effet COMPLET (volume + classes
     jamais vues) ;
   - F1 macro restreint aux classes PRÉSENTES dans le train → effet VOLUME pur ;
   - l'écart entre les deux = le coût des habitats jamais vus dans le train.
3. Interprétation pour le manuscrit : l'ancienne courbe par tuiles aléatoires
   (stratifiée, toutes classes présentes à chaque fraction — FRACS =
   0.01/0.05/0.10/0.25/0.50/0.70/1.00 dans `slurm_datacurve.sh`) donne des F1
   plus hauts aux petites fractions que la courbe spatiale. L'écart entre les
   deux courbes QUANTIFIE le confondant d'autocorrélation spatiale : c'est le
   résultat attendu, pas un artefact. La décomposition par la double métrique
   explique d'où vient l'écart (volume seul vs couverture d'habitats).

## Utilisation (phase d'entraînement : TERMINÉE — 21/21 runs, 2026-08-14)

Le pipeline d'entraînement existant (`scripts/datacurve_one_run.py`) entraîne sur
la TOTALITÉ d'un CSV en passant `--fraction 1.0` ; chaque split spatial est donc
un run `--fraction 1.0` sur `spatial_datacurve/splits/fracXXX_seed{S}/train.csv`.
Exécuté via `slurm_datacurve_spatial_v2.sh` (21 runs, y compris la relance de
`frac100_seed2` au-delà du mur de 6 h). Résultats : voir `RESULTS_REPORT.md` et
`lora_spatial_v2/results_spatial_summary.csv` ; intégrés à
`rapport_bouguessa/datacurves.pdf` §3.
