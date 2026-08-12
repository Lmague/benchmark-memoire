# Annotateur local d'orthomosaïques

Outil local d'annotation ponctuelle sur orthomosaïques géoréférencées volumineuses
(8 à 16 Go par fichier), sur une machine à mémoire limitée. Serveur Python
(FastAPI) + frontend HTML/JS léger, lancé en une commande et utilisé dans le
navigateur. Toutes les annotations de toutes les sessions vont dans **un seul
GeoPackage de sortie**, structuré comme le jeu d'annotations existant (une couche
par espèce, colonnes `['Label', 'geometry']`, géométrie `Point`, EPSG:32618).

S'y ajoute une **prospection zero-shot** : au lieu de parcourir 200 000 tuiles à
la main, l'outil propose les endroits où il pense qu'il y a une plante, et il
mesure lui-même à quel point il a raison, espèce par espèce.

## Fonctionnalités

- Navigation par tuiles avec **minicarte** (vue d'ensemble cliquable) et saut
  automatique des **tuiles vides** (bordures de remplissage de l'orthomosaïque).
  La minicarte affiche les points annotés (les miens en couleur d'espèce, les
  existants en petits points clairs) et les **candidats de prospection**.
- **Changement d'orthomosaïque depuis l'interface**, sans changer de fichier de sortie.
- Deux modes : pose de **points** par espèce, ou tracé de **zones « à revoir »**.
- **Galerie de référence** par espèce à deux niveaux : vues du dessus extraites du
  dataset (même point de vue que l'annotation) et photos iNaturalist au sol, avec
  comparaison côte à côte.
- **Références affichées à la même échelle que la tuile** : une vignette de 1,2 m
  occupe à l'écran exactement la largeur que 1,2 m de terrain occupent dans la
  vue courante, et elle se redimensionne au zoom. Sans cela, comparer deux formes
  vues du dessus induit en erreur. Une seule espèce est développée à la fois —
  celle qui est active, survolée, ou celle du candidat sous le curseur — car à la
  vraie échelle six espèces feraient un panneau de 1300 px de haut.
- **Loupe** : la vignette de 1,2 m située sous le curseur s'affiche en haut du
  panneau de gauche, juste au-dessus des références — comparaison immédiate.
- **Prospection** : candidats affichés sur la tuile, acceptables **d'un clic**,
  navigation « candidat suivant » par score décroissant, analyse à la demande de
  la tuile affichée (~1 s).
- Calque des annotations existantes, zoom, légende, annulation multi-niveaux,
  sauvegarde incrémentale.

## Garanties clés

- **Jamais de raster chargé en entier.** Toute lecture pixel passe par
  `rasterio.windows.Window` + `src.read(window=...)`, bornée à la tuile affichée
  ou au bloc traité. Les balayages complets se font en blocs, à mémoire bornée,
  sur les **overviews internes** des fichiers.
- **Un seul fichier de sortie**, mis à jour de façon incrémentale (append) après
  chaque action.
- **Conversion pixel → UTM déléguée à rasterio** (`rasterio.transform.xy` /
  `rowcol`), jamais recodée à la main. Validée avant toute annotation réelle.
- **Sources en lecture seule stricte.** `Dataset_Leo/` n'est jamais écrit ;
  `Annotations.gpkg` / `Annotations_polyg.gpkg` ne sont jamais modifiés ni recopiés.
  L'outil refuse toute sortie portant ces noms ou située sous `Dataset_Leo/`.
- **Aucun accès réseau à l'exécution.** Les poids des encodeurs sont lus dans le
  cache Hugging Face local (`HF_HUB_OFFLINE=1` est forcé).

## Installation

```bash
pip install -r tools/ortho_annotator/requirements.txt
```

`torch`, `transformers`, `scipy` et `scikit-learn` ne sont nécessaires que pour la
prospection. Sans eux, l'annotateur fonctionne, et l'étage couleur aussi (il ne
demande que `scipy`).

## Lancement

Depuis `tools/ortho_annotator/` :

```bash
python -m ortho_annotator serve \
  --raster "/home/erazal/Documents/Mémoire/Dataset_Leo/Orthomosaiques/Orthom_Clairiere_9Aout23_WGS84UTM18N.tif" \
  --output "$HOME/annotations_leo/session.gpkg"
```

Puis ouvrir http://127.0.0.1:8000/.

Le même `--output` doit être réutilisé pour toutes les sessions et toutes les
orthomosaïques : c'est le fichier unique partagé.

### Lancement en app (raccourci clavier)

`scripts/launch.sh` évite de retaper la commande : il mémorise la dernière
orthomosaïque choisie (dialogue de sélection de fichier au premier lancement),
démarre le serveur s'il ne tourne pas déjà, et ouvre le navigateur sur
`--output` toujours fixé à `$HOME/annotations_leo/session.gpkg`.

Une entrée d'application est installée dans
`~/.local/share/applications/ortho-annotator.desktop` : appuyer sur **Super**
et taper `annotator` (ou `ortho`) fait apparaître « Ortho Annotator » dans le
lanceur, comme n'importe quelle application. Relancer alors que le serveur
tourne déjà se contente de rouvrir l'onglet.

Pour arrêter le serveur : `pkill -f "ortho_annotator serve"`.

### Paramètres (`serve`)

| Argument | Défaut | Rôle |
|---|---|---|
| `--raster` | *(requis)* | Chemin du `.tif` à annoter. |
| `--output` | *(requis)* | GeoPackage de sortie UNIQUE, partagé entre sessions. |
| `--tile-size-m` | `5.0` | Taille de tuile en **mètres**, convertie en pixels via la résolution réelle. |
| `--overlap-m` | `0.0` | Chevauchement entre tuiles, en mètres. |
| `--cache-size` | `9` | Nombre de tuiles gardées dans le cache LRU. |
| `--tile-format` | `jpeg` | `jpeg` (20× plus rapide à encoder) ou `png` (sans perte). |
| `--tile-quality` | `90` | Qualité JPEG. |
| `--existing-annotations` | `Dataset_Leo/.../Annotations.gpkg` | GeoPackage existant : source des noms d'espèces **et** calque (LECTURE SEULE). |
| `--reference-dir` | *(aucun)* | Dossier d'images de référence SUPPLÉMENTAIRE. |
| `--reference-map` | *(aucun)* | JSON optionnel `code_espèce -> nom de référence`. |
| `--panel-position` | `side` | Position du panneau de référence : `top` ou `side`. |
| `--no-prospect` | | Ne pas charger les modèles de prospection. |
| `--model` / `--threads` | *(auto)* / `4` | Encodeur figé et threads CPU. |
| `--host` / `--port` | `127.0.0.1` / `8000` | Adresse d'écoute. |

Les noms d'espèces et de colonnes ne sont **jamais inventés** : ils sont lus tels
quels dans le GeoPackage existant (ou dans le fichier de sortie déjà initialisé).

## Raccourcis clavier

| Touche | Action |
|---|---|
| `←` / `→` | Tuile précédente / suivante |
| `↑` / `↓` | Rangée de tuiles précédente / suivante |
| `N` / `P` | Tuile avec contenu suivante / précédente |
| `1`–`9` | Activer l'espèce correspondante |
| `0` ou `Échap` | Désactiver l'espèce active |
| `B` | Basculer « Points » / « Zone à revoir » |
| **`S`** | **Sauter au candidat de prospection suivant (score décroissant)** |
| **`A`** | **Analyser la tuile affichée** |
| **`C`** | **Afficher / masquer les candidats** |
| `Ctrl+Z` | Annuler la dernière action (≥ 20 niveaux) |
| `E` | Calque des annotations existantes |
| `+` / `-` | Zoomer / dézoomer |

Souris :

- **Clic gauche sur l'image** : pose un point (menu contextuel si aucune espèce active).
- **Clic droit sur un point posé** : le supprime.
- **Clic sur un candidat** (cercle pointillé jaune) : l'accepte et le transforme en
  annotation. **Clic droit** : l'écarte définitivement.
- **Glisser en mode « Zone à revoir »** : trace un rectangle.
- **Clic sur la minicarte** : saut direct vers la zone.

---

# Prospection : trouver les plantes sans modèle entraîné

Le jeu existant contient **10 757 points déjà annotés**. Ils ne servent pas qu'à
l'affichage : ce sont les exemples à partir desquels l'outil apprend à reconnaître
chaque espèce, **sans aucun entraînement de réseau** (pas de réglage fin, pas de
descente de gradient). Deux détecteurs complémentaires, choisis parce qu'ils
échouent sur des choses différentes.

## 1. Heuristique couleur apprise

À 2,4 mm/px et début août, ce qui distingue le plus une espèce vue du dessus,
c'est la couleur de son inflorescence. Plutôt que d'écrire des seuils HSV à la
main, on estime deux distributions de couleurs en CIE Lab — celle des pixels
autour des points annotés, celle du fond — et on note chaque pixel par le
**rapport de vraisemblance** :

```
score(couleur) = log p(couleur | espèce) − log p(couleur | fond)
```

Puis lissage à l'échelle d'une inflorescence, seuillage, composantes connexes,
filtrage par taille. Le score des 262 144 couleurs 6 bits est précalculé une fois,
si bien que noter un bloc revient à une indexation : le balayage d'une
orthomosaïque entière est limité par le disque, pas par le calcul.

**Point aveugle assumé** : une espèce défleurie a la couleur du fond. L'asclépiade
(*Asclepias syriaca*) a fini de fleurir le 9 août — mesuré, la couleur ne la
trouve pas (rappel 0,05).

## 2. Appariement dense de jetons (DINOv3, figé)

Pour ces cas-là il faut de la **texture et de la forme**. Un seul passage avant de
DINOv3 ViT-S/16 sur une fenêtre de 784×784 donne 49×49 jetons de patch en 0,8 s
sur ce CPU, soit une carte de descripteurs à ~10 cm de résolution au sol pour une
fenêtre de 5 m. Chaque jeton est comparé à une banque de jetons prélevés aux
points annotés :

```
score(jeton) = max cos(jeton, jetons de l'espèce) − max cos(jeton, jetons de fond)
```

Le terme de fond est ce qui rend le score discriminant : sans lui, tout ce qui est
végétal ressemble à tout ce qui est végétal.

Astuce qui rend la construction bon marché : les points annotés sont groupés, une
seule fenêtre de 5 m en contient souvent des dizaines. Un passage avant fournit
donc des dizaines de jetons de référence d'un coup, et tous les jetons éloignés de
toute annotation servent de jetons de fond.

## 3. Calibration mesurée, pas de seuils devinés

Aucun seuil n'est choisi à la main. Pour chaque espèce, chaque détecteur est
rejoué sur des fenêtres **réservées** (non utilisées pour construire les
références) et confronté aux points annotés ; on garde le seuil qui maximise le
F1. Résultat obtenu sur les 7 orthomosaïques (rappel / précision, appariement à
35 cm) :

| Espèce | Couleur (R/P) | Dense (R/P) | Détecteur retenu |
|---|---|---|---|
| `Ascsyr` | 0,05 / 0,18 | **0,63 / 0,35** | dense |
| `Daucar` | **0,97 / 0,60** | 0,57 / 0,50 | couleur |
| `Eumac`  | **0,94 / 0,66** | 0,63 / 0,50 | couleur |
| `Leuvul` | **0,80 / 0,72** | — | couleur |
| `Lotcorn`| 0,65 / 0,40 | **0,86 / 0,61** | dense |
| `Solcan` | 0,83 / 0,54 | **0,96 / 0,78** | dense |

Deux avertissements à garder en tête en lisant ces chiffres :

- **La précision est un minorant.** Les annotations existantes sont incomplètes ;
  une proposition comptée comme fausse est parfois une vraie plante que personne
  n'avait encore pointée — c'est précisément ce qu'on cherche.
- **Le rappel est mesuré à 35 cm** : « le détecteur a-t-il pointé à moins de 35 cm
  d'une plante connue », pas « a-t-il détouré la plante ».

`Leuvul` n'a que 23 points annotés, tous sur une seule orthomosaïque : ses chiffres
sont indicatifs. Le diagnostic est aussi utile que la détection : il dit sur
quelles espèces se reposer sur l'outil, et sur lesquelles rester à l'œil.

## Utilisation

```bash
# 1. Apprendre les détecteurs (une fois, ~10 min ; refaire si de nouvelles
#    annotations changent nettement la donne)
python -m ortho_annotator prospect learn --output "$HOME/annotations_leo/session.gpkg"

# 2. Balayer une orthomosaïque (hors ligne ; l'étage dense est le plus lent)
python -m ortho_annotator prospect scan \
  --output "$HOME/annotations_leo/session.gpkg" \
  --raster ".../Orthom_Clairiere_9Aout23_WGS84UTM18N.tif"

# 3. Annoter : les candidats apparaissent dans l'interface, `S` saute au suivant
python -m ortho_annotator serve --raster ... --output ...

# État des modèles et des candidats
python -m ortho_annotator prospect status --output "$HOME/annotations_leo/session.gpkg"

# Export des candidats pour QGIS
python -m ortho_annotator prospect export \
  --output "$HOME/annotations_leo/session.gpkg" --dest "$HOME/annotations_leo/candidats.gpkg"
```

Options utiles de `scan` : `--mode color|dense|auto` (défaut `auto`, suit la
calibration), `--species Solcan Daucar`, `--max-windows N` pour un essai rapide,
`--exclude-m` (distance en deçà de laquelle un candidat est jugé déjà annoté),
`--max-per-species`, `--no-rerank`.

Ordre de grandeur mesuré sur « Clairiere » (150 × 148 m, 58 366 × 57 342 px), CPU
4 threads : étage couleur 210 blocs en ~2 min, re-classement de 900 vignettes par
espèce en ~3 min, étage dense 549 fenêtres de 5 m (381 bordures sautées) en
~20 min. Les candidats couleur au-delà de `--rerank-top` sont **écartés** plutôt
que gardés avec un score couleur : mélanger deux échelles de score rendrait le
classement « candidat suivant » incohérent.

**L'outil ne propose que du nouveau** : tout candidat à moins de 40 cm d'un point
déjà annoté (jeu existant ou vos propres poses) est écarté avant affichage. Et un
candidat que vous écartez d'un clic droit ne réapparaît pas au balayage suivant.

Les artefacts sont écrits dans `<dossier du GeoPackage de sortie>/prospect/` :
`colormodel.npz`, `dense_tokens.npz`, `prototypes.npz`, `calibration.json`,
`candidates.sqlite`. Jamais sous `Dataset_Leo/`.

## Vignettes de référence

```bash
python -m ortho_annotator rebuild-crops --crop-m 1.2 --per-species 16
```

Redécoupe les vues du dessus aux points annotés et **enregistre l'emprise au sol**
dans `reference_crops/meta.json` — c'est cette valeur qui permet à l'interface
d'afficher les références à la même échelle que la tuile.

---

## Export secondaire « à plat » (optionnel)

```bash
python -m ortho_annotator export-flat \
  --source "$HOME/annotations_leo/session.gpkg" \
  --dest   "$HOME/annotations_leo/session_flat.gpkg" --layer annotations
```

## Tests

```bash
cd tools/ortho_annotator
python -m unittest tests.test_validation tests.test_prospect -v
```

`test_prospect` fabrique un GeoTIFF synthétique : il vérifie la conversion Lab, la
détection des taches, l'exclusion des points déjà connus, la persistance des
candidats et les décisions qui survivent à un nouveau balayage — sans toucher aux
données réelles.

Validation ciblée sur un raster réel :

```bash
python -m ortho_annotator validate --raster <chemin.tif>
```

La validation des coordonnées (coins + centre comparés à `src.bounds` à la
tolérance d'un demi-pixel) est **rejouée automatiquement au lancement** de `serve`
et bloque le démarrage en cas d'échec.

---

## Notes de performance (mesurées sur ces fichiers)

Les GeoTIFF de `Dataset_Leo` sont **en bandes** (`blockysize=32`, pas de tuilage
interne) : lire une fenêtre de 2000 px de haut oblige GDAL à décompresser une
soixantaine de bandes pleine largeur. D'où les choix suivants.

| Point | Avant | Après |
|---|---|---|
| Encodage d'une tuile de 1932² | PNG, 292 ms, 10 Mo | JPEG q90, **15 ms, 2,2 Mo** |
| Calque des annotations existantes | 6 `gpd.read_file(bbox=…)` **par tuile** | index numpy chargé une fois, **3 ms** |
| Préchargement | 8 voisins, sous verrou global | 3 voisins **dans le sens du parcours**, thread dédié, annulable |
| Accès concurrents au raster | un verrou global | un handle rasterio **par thread** |
| Cache GDAL | 5 % de la RAM (défaut) | plafonné à 256 Mo |

Mesures de bout en bout : tuile servie en 0,21 s (0,12 s si préchargée), loupe de
1,2 m en 0,02 s, analyse d'une tuile en 0,4 s une fois torch chargé.

---

## Rapport : VÉRIFIÉ vs. supposé

Tout ce qui suit a été **lu sur les fichiers réels** (`rasterio.open`,
`fiona.listlayers`, `geopandas.read_file`, `sqlite_master`, `ls`), non supposé.

### Orthomosaïques (`Dataset_Leo/Orthomosaiques/`, GeoTIFF)

Toutes vérifiées : **EPSG:32618**, **3 bandes RGB `uint8`**, compression **LZW**,
**aucune valeur `nodata`**, masque `all_valid`, **overviews internes** présentes
(facteurs 2 à 512), stockage **en bandes** de 32 lignes.

| Fichier | Dimensions (px) | Résolution (m/px) |
|---|---|---|
| `Orthom_Maison_9Aout23_WGS84UTM18N.tif` | 123902 × 52222 | 0,00241116 |
| `Orthom_Clairiere_9Aout23_WGS84UTM18N.tif` | 58366 × 57342 | 0,00258767 |
| `Orthom_TrailErable_9Aout23_WGS84UTM18N.tif` | 72704 × 93184 | 0,00233568 |
| `Orthom_Rousseau_..._0-0 / 0-1 / 1-0 / 1-1.tif` | 74751 × 59391 | 0,00230474 |

### GeoPackage existant (`Annotations.gpkg`)

6 couches : `Lotcorn`, `Leuvul`, `Ascsyr`, `Daucar`, `Eumac`, `Solcan`. Schéma
Fiona `Label: str:255`, géométrie `Point`, EPSG:32618. Effectifs : 307 / 23 /
1050 / 1106 / 3351 / 4920. Répartition vérifiée par orthomosaïque : `Lotcorn` et
`Leuvul` **uniquement** sur « Maison », `Daucar` surtout sur « TrailErable »,
`Eumac` et `Solcan` surtout sur « Maison ». C'est pourquoi les détecteurs sont
appris en **mutualisant toutes les orthomosaïques** : un modèle appris sur un
seul vol serait aveugle à la moitié des espèces.

### Environnement Python (vérifié)

Présents : FastAPI 0.139, Uvicorn 0.51, rasterio 1.5, Fiona 1.10.1, GeoPandas
1.1.3, Shapely 2.1.2, Pillow 12.1, pyogrio 0.12.1, torch 2.12 (**CPU seulement**,
pas de GPU sur cette machine), transformers 5.5.4, scipy 1.17, scikit-learn 1.8.
**Absents** : Flask, `osgeo`, pytest, timm, open_clip, OpenCV, scikit-image.
Conséquences assumées : stack **FastAPI** ; tests en **unittest** ; les fonctions
SQL GeoPackage requises par les déclencheurs rtree d'OGR sont **réenregistrées
côté `sqlite3`** ; la détection dense utilise `scipy.ndimage` (pas OpenCV).

Poids présents dans le cache Hugging Face local : `dinov3-vits16`, `dinov3-vitb16`,
`dinov3-vitl16` (`pretrain-lvd1689m`). Le ViT-S/16 est utilisé par défaut : 21,6 M
paramètres, ~19 vignettes/s en 224² sur 4 threads, 0,77 s pour un passage dense en
784². Aucun téléchargement.

### Choix d'implémentation (non dictés par les données)

- Le fichier de sortie est **créé** via Fiona puis **muté** via `sqlite3` pour
  l'append/suppression incrémentaux. Les couches existantes ne sont jamais réécrites.
- Les candidats sont stockés en **SQLite simple** (pas en GeoPackage) : ce sont des
  fichiers de travail, réécrits à chaque balayage ; `prospect export` produit le
  GeoPackage quand on veut les ouvrir dans QGIS.
- Résolution des histogrammes couleur (12×20×20 en Lab) et taille de fenêtre dense
  (5 m, 784 px) : choix de compromis, ajustables en ligne de commande.
- Palette de 6 couleurs distinctes assignée par ordre d'espèce.
