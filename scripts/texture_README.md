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

## Sur Narval (le seul endroit où les tuiles existent)

### Dépendances Python

La texture n'utilise **que numpy, scipy et PyWavelets** — `src/texture.py` et la sonde
n'importent ni torch ni aucun backbone. Ne pas lancer `pip install -r requirements.txt` :
ce fichier exige `torch>=2.4`, absent du wheelhouse Alliance, et l'installation échoue pour
rien.

```bash
source ~/ENV/bin/activate
python -c "import numpy, scipy, pywt; print('OK')"      # le job fait ce test lui-même
pip install PyWavelets                                    # seulement si pywt manque
```

Le job affiche l'interpréteur utilisé et les versions, puis la commande d'installation
exacte pour les seuls modules manquants :

```
[slurm] interpréteur : /home/lmague/ENV/bin/python
[slurm] dépendances OK — numpy 1.26.4, scipy 1.11.4, pywt 1.5.0
```

### Pré-requis sur `$SCRATCH`

| Chemin | Contenu |
|---|---|
| `tiles.zip` | tuiles 224 px (dézippées dans `$SLURM_TMPDIR`) |
| `splits_11cls/{train,val,test}.csv` | schéma **remappé 11 classes** — convention des configs SOTA (`csv_dir: ${SCRATCH}/splits_11cls`, chargé en `{split}.csv`) |
| `embeddings/<modèle>_{train,val,test}{,_labels}.npy` | embeddings canoniques |

Le script accepte aussi `splits/<split>_11cls.csv` (suffixe, sortie par défaut de
`scripts/generate_splits_11cls.py`) et teste les deux dans cet ordre. Il ne retombe **jamais**
sur `splits/<split>.csv` : c'est le schéma **brut 12 classes**, dont `train.csv` compte
49 433 lignes (RHOL incluse) contre 49 281 dans les embeddings — un cache extrait dessus
serait désaligné et l'ablation porterait sur les mauvaises tuiles.

Si aucun des deux n'existe :

```bash
python3 scripts/generate_splits_11cls.py --splits-dir "$SCRATCH/splits" \
    --out-dir "$SCRATCH/splits_11cls" --suffix ""
```

**Vérification d'alignement automatique.** Avant d'extraire, le job compare le nombre de
lignes de chaque CSV au nombre de lignes d'embedding **effectivement utilisées** :

```
[slurm] alignement train: csv=49281  embeddings=49281 (49433 bruts, RHOL retirée)  OK
[slurm] alignement val: csv=13209  embeddings=13209  OK
[slurm] alignement test: csv=17598  embeddings=17598  OK
```

Le « effectivement » compte : les embeddings gelés comptent **49 433** lignes sur train car
la tuile RHOL y est **intercalée** (positions 16 764…, 152 tuiles). Le chargeur les retire
avant la sonde, ce qui redonne exactement les 49 281 lignes de `splits_11cls/train.csv` —
vérifié élément par élément, **0 désaccord**. Comparer au nombre brut de lignes déclencherait
une fausse alerte sur train, et seulement sur train (val et test n'ont aucune tuile RHOL).

Un désalignement (mauvais CSV, ou mauvais schéma dans `MODELS_SPEC`) annule l'extraction avant
qu'elle ne tourne. C'est le garde-fou qui a rattrapé l'erreur de chemin de la première
soumission et la confusion 11cls/12cls de la seconde.

### Lancement nominal

```bash
# sur Narval
ssh narval.alliancecan.ca
cd ~/benchmark-memoire && git pull          # récupérer le commit texture
mkdir -p logs                                # une seule fois (logs/ n'est pas versionné)
sbatch scripts/slurm_texture.sh              # éditer --account dans l'en-tête avant !
```

Le job fait tout : dézippe `$SCRATCH/tiles.zip` dans `$SLURM_TMPDIR`, extrait les 3 splits,
lance le contrôle de non-régression, puis l'ablation des modèles listés dans `MODELS_SPEC`.
Il est **repartable** : relancer la même commande après un time limit reprend où il s'était
arrêté (chunks d'extraction sautés, modèles dont le JSON existe sautés).

### Rapatriement

Le job affiche la commande en fin de log ; la voici :

```bash
# depuis la machine locale
rsync -avP <user>@narval.alliancecan.ca:$SCRATCH/texture/ results/texture/
```

### Avant de dépenser 8 h : le dry-run d'extraction (1 min)

```bash
SPLITS=test SKIP_ABLATION=1 sbatch scripts/slurm_texture.sh
```
Extrait seulement le split test et s'arrête. Vérifie dans le log :

- `[slurm] CSV test : .../splits_11cls/test.csv  [splits_11cls/ (configs SOTA)]` — le bon schéma ;
- `[slurm] alignement test: csv=17598  embeddings=17598  OK` — l'ordre des tuiles ;
- `cache test: [17598, 106]  non-finis=0` — 0 tuile illisible.

### Ajouter des modèles

`MODELS_SPEC` est une liste `<spec>:<schéma>`, une par ligne. `<spec>` accepte les **deux
conventions d'embeddings du dépôt**, auto-détectées :

| Convention | Chemin | Source |
|---|---|---|
| **gelé** | `<spec>_<split>.npy` — ex. `embeddings/simdinov2_vitb16_train.npy` | `load_features` |
| **affiné** | `<spec>/<split>.npy` — ex. `<run_dir>/train.npy` | `load_sota_features` |

```bash
export MODELS_SPEC="$SCRATCH/embeddings/simdinov2_vitb16:12cls
$SCRATCH/embeddings/dinov3_vitb16_lvd:12cls
$SCRATCH/ft_ssl_results/dinov3_vitb16_lvd_lora_runs/dinov3_vitb16_lvd_lora_frac100_seed0:11cls"
sbatch scripts/slurm_texture.sh
```

Gelé → `12cls`. Run affiné → `11cls`. Déclarer le mauvais schéma n'est pas silencieux : la
vérification d'alignement le détecte (49 433 vs 49 281 sur train) et annule l'extraction.

### Ablations seules, sur un cache déjà extrait

```bash
SKIP_EXTRACTION=1 sbatch scripts/slurm_texture.sh
```
Utile pour ajouter un modèle : l'extraction (et `tiles.zip`) est alors entièrement évitée.

## Coût mesuré

| Étape | Coût |
|---|---|
| Extraction | ~21 ms/tuile en mono-processus, **~1 min par 1000 tuiles** à 4 processus ; 80 088 tuiles ≈ **5 min** à 16 processus |
| Contrôle de non-régression | ~4 min par modèle |
| Ablation complète (baseline + 7 familles + combinée) | **46 min 47 s mesuré** pour un modèle (126 ajustements lbfgs mono-thread sur 49 281 × 768–874) |

Une entrée de `MODELS_SPEC` ≈ **1 h**. La limite de 8 h du job couvre 8 modèles.

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
- **Contrôle de non-régression** : baseline SimDINOv2-B = 0,4719 (11cls, registre 0,4723)
  et **0,6534** (8cls sonde séparée, canonique 0,6537) — écart 0,0003–0,0004, dans la
  tolérance de reprobe.
- **Test de falsifiabilité** (le plus important) : ablation complète sur un cache de texture
  **purement aléatoire**. Aucune famille ne fabrique de gain — les 8 Δ sont dans ±0,001 :

  | famille | cols | Δ11cls | IC95 | p |
  |---|---|---|---|---|
  | glcm | 30 | −0,0003 | [−0,0019, +0,0013] | 0,70 |
  | glrlm | 16 | +0,0002 | [−0,0012, +0,0017] | 0,87 |
  | glszm | 16 | −0,0001 | [−0,0017, +0,0013] | 0,85 |
  | gldm | 15 | −0,0003 | [−0,0014, +0,0007] | 0,60 |
  | ngtdm | 5 | +0,0006 | [−0,0003, +0,0016] | 0,21 |
  | dwt | 10 | −0,0005 | [−0,0017, +0,0006] | 0,35 |
  | fourier | 14 | +0,0006 | [−0,0006, +0,0019] | 0,33 |
  | combined | 106 | −0,0010 | [−0,0031, +0,0011] | 0,37 |

  et `best_C` reste identique (0,001) sur les 9 jeux : ajouter 106 colonnes de bruit ne
deplace ni la régularisation ni le F1. **Test inverse** : un signal planté dans 3 colonnes
est détecté (Δ11 = +0,0081, p = 0,000).
- **Dry-run du `.sh` complet** : dézippage, extraction, résumé de cache, contrôle, ablation,
  rapatriement — sur un `tiles.zip` réduit à 120 tuiles.
- **Robustesse aux tuiles manquantes** : par défaut une tuile illisible produit une ligne
  NaN (comptée, listée dans le JSON, et la sonde **refuse** ensuite le cache plutôt que de
  propager des NaN via `StandardScaler`) ; `--strict` rétablit l'arrêt immédiat.
