# Texture — descripteurs classiques concaténés à l'embedding

*Ajouté 2026-09. Scripts : `scripts/texture_features.py` (extraction), `scripts/texture_ablation.py`
(sonde + bootstrap apparié). Module : `src/texture.py`. Tests : `tests/test_texture_features.py`.*

## Pourquoi

Arctic-TVC est à ~2,2 mm/px. Le patch d'un ViT (16 px) couvre **3,5 cm** : la texture fine —
moucheture du lichen, tapis de mousse, linaires de linaigrette — est lissée avant d'entrer
dans le réseau. Les descripteurs de texture classiques sont l'outil standard de la
télédétection de végétation pour cette échelle (Kulich et al. 2026,
`doi:10.3389/fpls.2026.1841696` : GLCM + GLDM + GLSZM, 135 features, XGBoost par pixel,
F1 macro 79 % ; Deng et al. 2022, `doi:10.1038/s41598-022-17620-2` : +15 % de F1 après
ajout de features de texture).

## Ce que ça n'est PAS

Ces features ne doivent **jamais** être empilées en canaux d'entrée d'un ViT gelé. Un patch
embedding attend 3 canaux normalisés : lui donner une pile `224×224×(x+3)` casse les filtres
pré-entraînés (Clay-CNN Hybrids 2026, `arXiv:2606.14081`, exclut explicitement DEM et pente
pour cette raison). La seule façon correcte de faire entrer des canaux supplémentaires dans
un ViT est un PEFT dédié (DEFLECT/UPE, `arXiv:2504.17397` : patch embedding séparé pour les
canaux neufs + attention non-entanglée).

Ici, la texture est **concaténée au vecteur d'embedding, avant la sonde**. C'est le même
mécanisme que la fusion contexte du R2 (768 tuiles + 768 contexte), déjà validé.

## Les 7 familles (106 features par défaut)

| Famille | Unité comptée | Sensible à | Rotation | Référence |
|---|---|---|---|---|
| `glcm` (30) | paires de pixels à un décalage | transitions, contraste | non | Haralick 1973 |
| `glrlm` (16) | segments (*runs*) | structures **linéaires** | non | Galloway 1975 |
| `glszm` (16) | zones connexes de même niveau | **taille des taches** | oui | Thibault 2009 |
| `gldm` (15) | voisins dépendants (\|Δ\| ≤ δ) | dépendance locale | oui | Sun & Wee 1983 |
| `ngtdm` (5) | écart au niveau moyen du voisinage | coarseness, busyness | oui | Amadasun & King 1989 |
| `dwt` (10) | énergie des sous-bandes ondelettes | échelle **et** orientation | oui | — |
| `fourier` (14) | spectre de puissance | pente β, **orientation** | oui | — |

Les préfixes sont la clé de l'ablation : `family_of("glszm_lae") == "glszm"`.

**Elles ne sont pas redondantes.** GLSZM est invariante à la rotation par construction,
alors que GLCM est directionnelle et qu'on en moyenne les 4 directions (perte
d'information). GLRLM voit les structures allongées (SEDG, TUSS) que ni GLCM ni GLSZM ne
capturent. D'où la table d'ablation famille par famille, puis combinée.

## Pourquoi pas PyRadiomics

L'implémentation de référence (conforme IBSI) est PyRadiomics, mais elle **ne s'installe pas
sur la machine de dev** (`metadata-generation-failed`, échec de compilation) et ajoute une
dépendance C fragile. `src/texture.py` réimplémente les matrices en numpy/scipy pur :
déterministe, sans compilation, testable en CI. Les définitions suivent PyRadiomics/IBSI
(à une exception documentée près : le seuil de lissage NGTDM `_EPS_NGTDM = 1e-6`).

## ⚠️ Deux définitions du « 8 classes » — ne jamais les mélanger

Constaté en codant l'ablation, et systématique sur les 9 modèles gelés où les deux sont
calculables :

| Définition | Comment | Où | Écart |
|---|---|---|---|
| **1. Sonde séparée** (canonique) | retire les tuiles ARCA/DRYI/RUBC (49281→48473 train, 17598→17277 test), compacte les labels en 0..7, **ré-ajuste** la sonde | `results/relance2/relance2_8cls.json`, `results/8cls/probe_knn_cgrid.json`, AGENTS.md §3 | référence |
| **2. Ré-moyenne** | garde la sonde 11 classes, ne moyenne que le F1 par classe des 8 classes restantes | champ `f1_macro_8cls_test` des `metrics.json` de FT (`scripts/datacurve_one_run.py:177`, `scripts/context_distill.py:576`) | **−0,0044 à −0,0063** |

Exemple mesuré (SimDINOv2-B) : sonde séparée **0,6534** (reproduit le canonique 0,6537 à
0,0003), ré-moyenne **0,6488**. L'écart (≈ 0,005) vaut **60 % de l'écart-type inter-seed
(0,008)** : un tableau qui compare des lignes issues des deux sources peut inverser des
paires proches. `scripts/texture_ablation.py` reporte les deux colonnes
(`f1_macro_8cls_sep`, `f1_macro_8cls_remoy`) pour rendre la distinction visible.

**Conséquence pour les tableaux existants** : toute table 8 classes qui combine des lignes
gelées (source 1) et des lignes affinées issues des `metrics.json` (source 2) est biaisée
d'environ +0,005 en faveur des lignes gelées. C'est le cas des tableaux 8 classes produits
avant cette note — à reprendre pour publication.

## Usage

```bash
# 1. Extraction — À LANCER LÀ OÙ LES TUILES EXISTENT (Narval : $SLURM_TMPDIR/tiles)
python3 scripts/texture_features.py --split train --n-jobs 8
python3 scripts/texture_features.py --split val   --n-jobs 8
python3 scripts/texture_features.py --split test  --n-jobs 8
#      → results/texture/{train,val,test}.npy + .json

# 2. Ablation — une famille à la fois, puis combinée
python3 scripts/texture_ablation.py --emb embeddings/simdinov2_vitb16 \
    --label-schema 12cls --tag simdinov2_vitb16 --include-combined
#      → results/texture/ablation_simdinov2_vitb16.json

# 3. Contrôle de non-régression : la baseline doit reproduire le F1 du registre
python3 scripts/texture_ablation.py --emb embeddings/simdinov2_vitb16 \
    --label-schema 12cls --baseline-only --tag controle
```

Coût mesuré : ~21 ms/tuile en extraction (85 tuiles/s à 4 processus), ~20 s par
ajustement de sonde sur 49 281 × 768 (mono-thread BLAS obligatoire, cf. AGENTS.md §4.8).

## Blocage connu (2026-09) : les tuiles natives sont absentes de la machine de dev

`out/tiles` **n'existe pas** localement ; les orthomosaïques Arctic-TVC non plus (seul
`Dataset_Leo/` — un autre jeu, du 9 août — est présent). Ce qui est disponible :
`out/context/context_{512,2048}/` = 30 807 images de **224×224** couvrant **100 % de val et
test** mais **0 % du train**, et ce sont les fenêtres de contexte **redimensionnées**
(512 px natifs → 224, soit un GSD effectif de ~5,0 mm/px, 2,3× plus grossier que la tuile).

Conséquences :
- l'ablation **ne peut pas** être calculée localement (pas de train) ;
- l'extraction sur les crops de contexte est possible mais mesure la texture à une autre
  échelle → **ne pas écrire ces caches dans `results/texture/`**, ils ne sont pas comparables
  à une extraction sur tuiles natives.

L'extraction réelle doit tourner sur Narval (`$SLURM_TMPDIR/tiles`, monté depuis `tiles.zip`)
ou là où `out/tiles` est reconstruit par `scripts/tilerization.py`.

## Validation effectuée

- **20 tests unitaires** (`tests/test_texture_features.py`) sur images synthétiques : contrat
  de colonnes, déterminisme, cas limites GLCM (uniforme dégénéré), ordres GLSZM (grandes vs
  petites zones), GLRLM (segments longs pour des bandes), GLDM (dépendance qui chute avec le
  bruit), normalisation des énergies DWT et des profils de Fourier, sens de β.
- **Plomberie de bout en bout** sur un cache de texture synthétique, avec `simdinov2_vitb16` :
  - cache **bruit pur** → Δ11 = −0,0003, IC95 [−0,0019, +0,0012], p = 0,59 (aucun gain) ;
  - cache avec **signal planté** dans 3 colonnes → Δ11 = +0,0081, p = 0,000.
- **Contrôle de non-régression** : baseline SimDINOv2-B = 0,4719 (11cls, registre 0,4723)
  et 0,6534 (8cls séparé, canonique 0,6537).
