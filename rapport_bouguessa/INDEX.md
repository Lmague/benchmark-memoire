# 📑 DOSSIER RAPPORT BOUGUESSA — INDEX

## Documents

| # | Fichier | Contenu | Pages |
|---|---------|---------|-------|
| 0 | `compendium.pdf` | **Toutes les données du projet en un document** — protocole, 21 modèles, courbes de données, géométrie, transférabilité, significativité + 4 annexes (111 runs bruts, F1 par classe, figures d'archive, inventaire des sources) | 30 |
| 1 | `performances.pdf` | **Tableau maître 21 modèles** — classement avec σ et IC95, F1 par classe, matrices de confusion, k-NN vs probe, LogME, significativité | 13 |
| 2 | `geometrie.pdf` | **Géométrie latente** — conventions de calcul, 21 modèles × 13 métriques, corrélations sur 3 populations, spectres, profondeur, famille DINOv3-B | 11 |
| 3 | `datacurves.pdf` | **Courbes de données** — 4 régimes × 8 fractions × 3 seeds, seuils de rentabilité, few-shot, par classe, géométrie vs volume à *n* fixe | 10 |
| 4 | `analyse.pdf` | **Analyse et interprétation** — le compagnon du compendium (qui s'interdit d'interpréter) : 7 constats, tableau maître, régimes, palier PEFT plat, contexte, géométrie, significativité, et ce que ça veut dire pour le mémoire | 17 |

## Fichiers sources

| Chemin | Rôle |
|--------|------|
| `*.tex` | Sources LaTeX des 5 documents |
| `tables/` | 30 fragments de tableaux, **générés** par `scripts/rapport/make_tables.py` |
| `figs/` | 22 figures PNG + les 3 scripts qui les produisent + `vizstyle.py` (palette) |
| `INDEX.md` | Ce fichier |

Aucun chiffre n'est saisi à la main dans les `.tex` : tous les tableaux sont des
`\input{tables/…}` régénérables.

## Pour compiler

```bash
# 0. une fois, si de nouveaux régimes LoRA sont ajoutés (calcule le F1 canonique
#    et le fusionne dans results/all_models_canonical_merged.json)
python3 scripts/probe_lora3_new_runs.py

# 1. données (à refaire seulement si les runs/embeddings changent)
python3 scripts/rapport/aggregate_screening.py
python3 scripts/rapport/geometry_test_fixed_n.py     # ~17 min
python3 scripts/rapport/knn_and_confusion.py
python3 scripts/rapport/layerwise_geometry.py
python3 scripts/rapport/correlations.py
python3 scripts/rapport/tier_ranking.py
python3 scripts/rapport/significance_tier.py   # ~1 h 30, cache disque
python3 scripts/rapport/check_consistency.py

# 2. tableaux + figures
python3 scripts/rapport/make_tables.py
for f in rapport_bouguessa/figs/make_*.py; do python3 "$f"; done

# 3. PDF (depuis rapport_bouguessa/, jamais depuis la racine)
cd rapport_bouguessa
for d in compendium performances geometrie datacurves; do
  pdflatex -interaction=nonstopmode $d.tex && pdflatex -interaction=nonstopmode $d.tex
done
```

## Résumé exécutif (30 secondes)

**Résultat central.** L'adaptation d'un ViT-B fonctionne, mais elle ne fait que
rattraper ce qu'un ViT-L gelé donne sans affinage. DINOv3-B LoRA r=8 bat son *propre*
backbone gelé de +0,0114, et l'écart survit à Benjamini-Hochberg sur les 91 paires du
palier (p = 0,0012) — ce régime n'avait jamais été testé formellement. Les deux autres
régimes du même backbone n'y arrivent pas (MHSA p = 0,016 ; Full p = 0,101). Le même
schéma se reproduit sur trois régimes LoRA r=8 ajoutés le 2026-07-28 (DINOv3-L,
SimDINOv2-B, SimDINOv2-L) : tous trois battent numériquement leur propre backbone gelé
(+0,0025 à +0,0080) sans qu'aucun écart ne passe le seuil après correction. Mais ce
gain ne dépasse pas la capacité : LoRA r=8 (DINOv3-B) contre DINOv3-L gelé donne
Δ = +0,0038, p = 0,39 ; les deux nouveaux régimes qui dépassent numériquement le
meilleur gelé (DINOv3-L et SimDINOv2-L LoRA, +0,0025 et +0,0044) ne sont pas non plus
significatifs (p = 0,55 et p = 0,60). Sur les 91 paires, 21 sont rejetées et **aucune
ne place un modèle affiné au-dessus du meilleur gelé**, tandis que DINOv3-L gelé en
bat quatre, dont son propre ViT-B gelé (+0,0076, p = 0,003).

**Méthodes.** 21 modèles : 9 gelés, 3 affinés depuis DINOv3-B, 1 depuis DINOv3-L,
3 depuis SimDINOv2-B, 1 depuis SimDINOv2-L, 3 depuis ViT-B/16 ImageNet, 1 ResNet-50
affiné. 135 runs de courbe de données
(4 régimes × 7–8 fractions × 3 seeds, plus le probing gelé). Probe canonique :
LogisticRegression lbfgs multinomial, grille C ∈ [1e-4, 10], `best_C` par validation,
`max_iter`=2000, seed 42, schéma 11 classes.

**Données.** Arctic-TVC, 12 classes de végétation arctique, imagerie drone
~2,2 mm/pixel, 38 orthomosaïques, 49 281 tuiles train / 13 209 val / 17 598 test,
découpage par orthomosaïque entière.

## Nouveautés de la révision 2026-07-25

Trois précautions de lecture sont documentées pour la première fois, avec les données
qui les établissent :

1. **Trois conventions de « rang effectif »** ont circulé sous le même nom, avec un
   écart d'un facteur ≈ 7 (entropie des valeurs singulières brutes / centrées /
   entropie des valeurs propres). Idem pour l'anisotropie (2 conventions) et NC1
   (2 conventions, de sens opposé). → `geometrie.pdf` §1.
2. **La géométrie mesurée sur le train confond richesse et taille d'échantillon** :
   le rang effectif est borné par min(n, D), donc il monte mécaniquement avec la
   fraction. Recalculée sur le jeu test à n fixe (17 598 tuiles), la montée
   disparaît pour tous les régimes pré-entraînés. → `datacurves.pdf` §7.
3. **Le tableau maître publié mélange deux protocoles** : colonnes spectrales
   calculées par seed puis moyennées, colonnes de clustering calculées sur les
   3 seeds concaténés. Pour un modèle affiné, concaténer trois modèles différents
   gonfle le rang effectif (LoRA r=8 : 43,7 → 71,6). → `geometrie.pdf` §1.

Ajouts de données : les runs de screening exploités au-delà du F1 global (F1 par
classe, exactitude, époque de convergence), géométrie à n fixe,
corrélations géométrie/F1 recalculées sur 3 populations avec IC95 bootstrap,
matrices de confusion, k-NN vs probe, géométrie par profondeur, régime few-shot.

## Révision 2026-07-26 — retrait des régimes ExPLoRA

Les deux régimes « ExPLoRA » (DINOv3-B ExPLoRA r=16 et ViT-B/16 IN-ExPLoRA) reposaient
sur une implémentation invalide : ils sont retirés de l'ensemble des documents. La
population passe de 20 à 18 modèles, la courbe de données de 5 à 4 régimes, et le
corpus d'affinage de 126 à 102 runs. Les runs et embeddings restent sur disque ;
l'exclusion est pilotée par `EXCLUDED_MODELS` dans `scripts/rapport/registry.py`.

Ce que le retrait déplace :

- les figures d'archive tracées sur l'ancienne population (manuscrit, `final_2026-07`)
  ne sont plus reproduites dans les documents.

Le bootstrap apparié ne contenait aucun ExPLoRA : le retrait ne change rien à la
différenciabilité statistique. C'est en le vérifiant qu'on a découvert qu'il ne
couvrait que 8 des 11 modèles du palier — voir la révision suivante.

## Révision 2026-07-26 — le palier compétitif passe à 11 modèles

En revérifiant le palier après le retrait, il est apparu que le « top-8 » n'était pas
une rupture dans les données mais une coupe de rang héritée de l'époque à 12 modèles
(`models_sorted[:8]`) : elle tombait entre deux modèles séparés de 0,0004. Le palier est
désormais `registry.TIER_K = 11`, la rupture mesurée — rangs 1 à 11 continus (étendue
0,0158), écart au rang 12 de +0,0057, près de trois fois l'écart médian entre rangs.
`registry.tier_break()` le recalcule et doit rester égal à `TIER_K`.

Ce que ça change dans les conclusions :

- **retiré** — « on peut classer les meilleurs modèles avec la séparabilité ». Son
  pouvoir d'ordonnancement décroît de façon monotone quand le palier s'élargit
  (Fisher : +0,89 à K=6, +0,69 à K=8, +0,28 à K=11) : artefact de petit échantillon.
  Sur le palier justifié, 60–62 % des 55 paires et p ≈ 0,37 ;
- **retiré** — la silhouette « significative dans les trois populations » : sur le
  palier son ρ tombe à +0,30 et son IC95 traverse zéro, comme celui de toutes les
  autres métriques ;
- **conservé et renforcé** — l'inversion des métriques spectrales, vraie pour *tous*
  les paliers testés (6, 8, 10, 11, 12) et redressée seulement sur les 18 modèles.
  C'est le résultat robuste de `geometrie.pdf` §4, et il est négatif.

## Révision 2026-07-26 — bootstrap apparié étendu au palier complet

La campagne publiée (`significance_matrix_8group_fresh.json`) ne testait que 8 des
11 modèles du palier. En étaient absents **le modèle n° 1** (DINOv3-B LoRA r=8) ainsi
que ViT-B/16 IN-Full et IN-MHSA — le tableau maître portait d'ailleurs la mention
« Non testé formellement ». Le résumé affirmait pourtant un tie statistique en citant
une table où LoRA r=8 ne figurait pas, et son ±0,0011 était un écart-type inter-seed,
pas un IC95 bootstrap (≈ 4 fois plus étroit).

`scripts/rapport/significance_tier.py` refait le test sur les 11 modèles (55 paires,
mêmes indices de rééchantillonnage que la campagne publiée). Contrôle de
non-régression : les 8 groupes communs se reproduisent à 2,7·10⁻⁴ près.

Ce que ça change :

- **LoRA r=8 bat significativement son propre backbone gelé** (+0,0114, p = 0,0012,
  rejeté par BH). Résultat nouveau — l'affinage n'est donc pas inutile ;
- **MHSA perd sa significativité** : p = 0,016, sous le seuil après correction sur
  55 paires au lieu de 28. La campagne à 8 groupes le donnait « significatif en
  limite » ;
- **aucun modèle affiné ne dépasse le meilleur gelé** : LoRA r=8 contre DINOv3-L
  donne p = 0,39. La conclusion « la capacité rapporte ce que l'adaptation
  rapporte » tient, et elle est maintenant testée pour le modèle n° 1.

Deux contrôles ajoutés à `check_consistency.py` : le bootstrap doit couvrir les
`TIER_K` modèles du palier, et ses 8 groupes communs doivent reproduire la campagne
publiée.

## Révision 2026-07-28 — ajout de 3 régimes LoRA r=8 (DINOv3-L, SimDINOv2-B, SimDINOv2-L)

Trois nouveaux régimes affinés (LoRA r=8, 3 seeds chacun) rejoignent les documents :
`dinov3_vitl16_lvd_lora`, `simdinov2_vitb16_lora`, `simdinov2_vitl16_lora`. La
population passe de 18 à 21 modèles, le corpus d'affinage de 102 à 111 runs (135 avec
le probing gelé). `scripts/probe_lora3_new_runs.py` calcule leur F1 canonique (même
protocole lbfgs multinomial, BLAS mono-thread) et le fusionne dans
`results/all_models_canonical_merged.json`.

Le palier compétitif passe de `TIER_K = 11` à `TIER_K = 14` (rupture recalculée par
`registry.tier_break()`) : les trois nouveaux régimes y entrent tous les trois sans en
déplacer aucun membre — les bornes du palier (0,4835 → 0,4677, étendue 0,0158) et
l'écart au rang suivant (+0,0057) sont inchangés, simplement décalés de 3 rangs.
`scripts/rapport/significance_tier.py` a été réexécuté avec les 3 nouveaux groupes
ajoutés à `GROUPS` (91 paires au lieu de 55) ; contrôle de non-régression : les 8
groupes historiques se reproduisent à 2,7·10⁻⁴ près (inchangé).

Ce que ça change dans les conclusions :

- **rien ne bascule** — le résultat central tient : DINOv3-B LoRA r=8 reste le seul
  régime qui bat significativement son propre backbone gelé (+0,0114, p = 0,0012).
  Les 3 nouveaux régimes battent tous numériquement leur propre gelé (+0,0025 à
  +0,0080) mais aucun ne passe le seuil après correction (SimDINOv2-B LoRA est le
  plus proche, p = 0,058) ;
- **deux nouveaux régimes dépassent numériquement le meilleur gelé** (DINOv3-L et
  SimDINOv2-L LoRA, +0,0025 et +0,0044 contre DINOv3-L gelé) mais ni l'un ni l'autre
  n'est significatif (p = 0,55 et p = 0,60) — la conclusion « aucun modèle affiné ne
  dépasse le meilleur gelé » se confirme sur deux backbones et deux familles de
  pré-entraînement supplémentaires plutôt que d'être infirmée ;
- **le pouvoir d'ordonnancement de la séparabilité n'est plus lisible comme une
  décroissance monotone** avec 21 modèles (ratio de Fisher : +0,32 à K=6, +0,75 à
  K=10, +0,39 à K=14, +0,30 à K=21 complet) — le signal culmine puis retombe, ce qui
  renforce (et précise) la lecture « artefact de petit échantillon » plutôt que de la
  contredire ;
- **l'inversion spectrale reste robuste pour le stable rank et le participation
  ratio** sur toutes les tailles de palier restreint (6 à 14) ; le rang effectif
  ($\sigma^2$) est plus nuancé (faiblement positif à K=6 seulement).

**Mise à jour :** 2026-07-28. `check_consistency.py` passe 13/13.

## Révision 2026-09-08 — compendium complet (42 modèles, palier à 35)

Le compendium absorbe tout le travail d'août-septembre, resté jusque-là hors des
documents (ou saisi à la main) :

- **Population 21 → 42 modèles** : +13 bras d'ablation LoRA/PEFT SimDINOv2-B
  (F1 canonique par reprobe mono-thread,
  `scripts/rapport/probe_simb_stageA_canonical.py`), +2 SimDINOv2-B entraînés
  Design B (metrics.json, même provenance que R1/R2/R3), +ViT-S/LoRA déjà au
  registre. Le palier compétitif passe de `TIER_K = 20` à **35** (rupture
  recalculée par `registry.tier_break()`, décrochage +0,0069 inchangé) ; le
  bootstrap apparié couvre 33 groupes (528 paires, cache disque réutilisé pour
  les 20 groupes historiques — non-régression à 2,7·10⁻⁴), les 2 entraînés
  SimB restant des point-estimates documentés (`NO_BOOTSTRAP_EMBEDDINGS`, pas
  d'embeddings rapatriés).
- **Nouveaux résultats scientifiques du recalcul** : (i) sur le palier à 35, la
  famille séparabilité passe le seuil BH (Fisher/CHI/NC1/DBI p ≤ 0,046,
  silhouette p = 0,026 en paires) — le verdict « artefact » à n=14 était un
  manque de puissance, pas une absence d'effet (direction stable sur tous les
  paliers) ; (ii) la spectrale reste anti-prédictive (41-42 %) ; (iii) b911/b611
  et NormTuning battent significativement le gelé SimB (p ≤ 0,0043) quand le
  r8a8 tous blocs n'y arrive pas (p = 0,058) ; (iv) R2 bat tout le monde
  (+0,025 à +0,029, p < 10⁻⁴) mais reste une borne non déployable ; aucun modèle
  déployable ne dépasse le meilleur gelé.
- **Nouveau chapitre « Adaptation paramétrique »** : ablations rang/blocs DINOv3
  (déjà dans performances.pdf), Stage A SimB (`t_simb_ablation` + 3 figures
  d'échelles), coût/performance, caveat de sélection du C (QKV seed2 : ±0,0074
  par basculement de C).
- **Nouveau chapitre « Contexte spatial »** (en corps de document, plus en annexe
  sauvage) : `t_ctx_distill` (R1/R2/R3 + 2 SimB entraînés), matrice d'attribution
  et contrôles Bouguessa **régénérés depuis les JSONs** (l'ancien fragment était
  saisi à la main), sweep frozen 5×3 (`t_ctx_sweep`), contrôles R2 **recalculés
  en local** (`scripts/context_bouguessa_controls.py`, 24 probes — reproduction
  des valeurs Narval à 0,0003 près), figure F1 étendue aux SimB entraînés.
- **Garde-fous ajoutés** : `check_consistency.py` gagne le contrôle d'équilibre
  des fragments (begin/end) et une couverture de palier par clés (plus par
  comptage) ; le régime `norm_tuning` est ajouté à `src/models.py`.

**Mise à jour :** 2026-09-08. `check_consistency.py` passe 15/15. Compendium :
51 pages. Les 3 autres documents compilent sans erreur (tables partagées
régénérées, prose inchangée — y compris `performances.tex` § cartes
d'activation, dont `make_actmaps.py` exige les tuiles brutes absentes du disque).
