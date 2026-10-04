# SSL aérien Léo × annotations — extraction, kNN/logreg/MLP et ensemble de seeds

Branche : dépôt Mémoire. Statut : **outillage livré et testé, aucun calcul réel lancé**
(les checkpoints SSL Léo ne sont pas encore rapatriés en local).

## 0. Environnement (à lire en premier)

Deux `base` cohabitent sur cette machine : **anaconda3** (Python 3.14, sans `torch` ni
`rasterio`) et **miniconda3** (Python 3.13, avec `torch`, `rasterio`, `geopandas`,
`sklearn`, `einops`). Tous les scripts locaux de ce dossier exigent **miniconda3** :

```bash
# soit l'interpréteur explicite (le plus simple)
/home/erazal/miniconda3/bin/python scripts/leo_ssl_annot_prepare_index.py --out ~/annot_index.npz

PY=/home/erazal/miniconda3/bin/python   # pour tout le reste de ce document

# soit activer le bon base
source /home/erazal/miniconda3/etc/profile.d/conda.sh
conda activate base          # py314 d'anaconda3 = MAUVAIS encore ; vérifier :
which python                 # doit afficher /home/erazal/miniconda3/bin/python
```

> `conda activate CV` échoue depuis anaconda3 (les envs de miniconda3 ne sont pas
> visibles) — c'est normal, passer par le `source .../miniconda3/etc/profile.d/conda.sh`.
> Les scripts affichent désormais un message explicite en cas de mauvais interpréteur.

## 1. La question

Le projet `annotations_leo/` encode ~6000 points annotés (5 espèces exploitables) de
l'ortho Clairière. Jusqu'ici, **seul le backbone SimDINOv2-B iNat gelé** a servi
d'encodeur (`embeddings/bank.npz`, `negatives.npz`, `grid_*.npy`). Les rapports
d'analyse concluaient : « levier d'amélioration : encodeur pré-entraîné sur de
l'aérien » (`embeddings/separability_report.md` §5.6).

Les 3 seeds ExPLoRA-Léo sont exactement ce levier. Ce banc mesure, **à folds spatiaux
appariés**, ce que l'adaptation aérienne change sur les annotations :

| | question | mesure |
|---|---|---|
| `multiclass` | les 5 espèces sont-elles mieux séparées ? | accuracy / F1-macro, folds identiques |
| `detection` | chaque espèce est-elle mieux détectée ? | AUPRC / ROC-AUC, négatifs identiques |
| `ensemble` | les 3 seeds moyennées apportent-elles plus qu'une seule ? | même folds, moyenne des probas |

## 2. Pipeline

### 2a. Local (CPU, raster brut) — contexte 512 disponible

```
local (CPU)                                    local (CPU)
───────────                                    ───────────
checkpoints/leo_ssl/seed{N}_last.pth
        │
        └─ bash scripts/leo_ssl_annot_extract.sh
              (annotations_leo/extract_embeddings.py --ckpt --out-dir)
           → embeddings/<tag>/{bank,negatives}.npz + contexte 512
                                              │
                                              ▼
         $PY scripts/leo_ssl_annot_bench.py --clf logreg,knn,mlp --variants tile,ctx,fused
```

### 2b. Narval (GPU, sans raster) — voie recommandée

Les 4 COG pèsent 131 Go et restent locaux, mais les **tuiles 224 JPEG sont déjà sur
Narval** (`$SCRATCH/web/tiles/clairiere`, 355 486, les mêmes que les embeddings gelés)
et le carroyage est **identique** (`idx = tr*260 + tc`, vérifié). On résout donc l'index
localement (quelques lectures d'en-tête) et on expédie 155 ko au lieu de 131 Go :

```
local (CPU, 30 s)                                Narval (GPU, ~2 min/tag)      Narval (CPU)
─────────────────                                ────────────────────────      ────────────
$PY scripts/leo_ssl_annot_prepare_index.py
    → annot_index.npz  (6037 pts + 4897 nég.)
        │  rsync (~3,5 Mo : index + manifest)
        ▼
$SCRATCH/annot_leo/annot_index.npz
$SCRATCH/web/tiles/clairiere/**/*.jpg
        │
        └─ sbatch scripts/slurm_leo_ssl_annot_eval.sh
              ├─ leo_ssl_annot_extract_web.py × 4 tags (base + 3 seeds)
              │      → $SCRATCH/annot_leo/emb/<tag>/{bank,negatives}.npz
              └─ leo_ssl_annot_bench.py  (apparié, variante tile)
                     → $SCRATCH/annot_leo/bench/{bench_annot.json,bench_annot.md}
```

> **Différence à connaître** : les JPEG web sont **stretchés** (percentiles 2-98 par
> tuile), le local lit le raster brut. Les embeddings des deux voies ne sont donc pas
> bit-comparables — mais la voie Narval ré-encode **aussi** la baseline, donc la
> comparaison gelé vs adapté reste valide. Autre conséquence : pas de contexte 512
> (les JPEG sont des tuiles 224 isolées ; recoller un voisinage serait incohérent vu le
> stretch par tuile). Le rapport de séparabilité a mesuré que le contexte n'apporte rien
> ici (0.878 fusionné vs 0.875 tuile) — la voie Narval se limite donc à `--variants tile`.

## 3. Commandes

### 3.0 Narval — préparer et expédier l'index (local, 30 s)

```bash
$PY scripts/leo_ssl_annot_prepare_index.py --out ~/annot_index.npz
# vérifie : "bank : 6037 points | 3319 tuiles uniques" + "rejet : 2933"

ssh narval 'mkdir -p $SCRATCH/annot_leo'
rsync -avP ~/annot_index.npz \
    ~/annotations_leo/web/manifest_clairiere.json \
    narval:$SCRATCH/annot_leo/
# (facultatif, provenance seulement) :
rsync -avP ~/annotations_leo/session.gpkg \
    ~/annotations_leo/prospect/candidates.sqlite \
    narval:$SCRATCH/annot_leo/
```

### 3.1 Local — extraction CPU + bench (contexte 512 disponible)

```bash
# rapatrier un checkpoint par seed (last.pth = le plus entraîné)
mkdir -p ~/Documents/Mémoire/checkpoints/leo_ssl
for s in 0 1 2; do
  rsync -avP narval:$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed${s}/checkpoints/last.pth \
      ~/Documents/Mémoire/checkpoints/leo_ssl/seed${s}_last.pth
done

PHASES=bank,negatives bash scripts/leo_ssl_annot_extract.sh     # ~1 h au total
# scan complet de l'ortho (optionnel, pour l'outil d'annotation) :
PHASES=grid bash scripts/leo_ssl_annot_extract.sh               # ~1,5-2 h par modèle

$PY scripts/leo_ssl_annot_bench.py --clf logreg,knn,mlp --variants tile,ctx,fused
```

`--ckpt` accepte un checkpoint ExPLoRA-Léo **tel quel** : format `teacher` + préfixe
`backbone.`, LoRA fusionnée — vérifié (`176/176 clés`).

> ⚠️ **Le `embeddings/bank.npz` historique est périmé** : il a été extrait le 2026-09-23 sur
> 4175 points, alors que `session.gpkg` en contient 6037 (vérifié le 2026-10-04). Le wrapper
> ré-encode donc **aussi** la baseline `base` — les `negatives.npz`, `grid_*.npy` et
> `detectors_final_v2.npz` existants sont dans le même état et doivent être régénérés pour
> rester cohérents avec le nouveau `bank.npz`.

### 3.2 Narval — job enchaîné après les 3 seeds SSL

```bash
# local : index (cf. 3.0) puis, sur Narval :
cd ~/benchmark-memoire && git pull && mkdir -p logs

SSLID=$(squeue -u $USER -n leo_explora_ssl -h -o %A | head -1)   # id du job array SSL
sbatch --dependency=afterok:$SSLID scripts/slurm_leo_ssl_annot_eval.sh

# une fois fini :
rsync -avP narval:$SCRATCH/annot_leo/bench/ results/leo_explora_ssl/annot_bench/
```

### Étape 3 — outil d'annotation (optionnel)

```bash
$PY scripts/leo_ssl_annot_bench.py --clf logreg --save-detectors --score-grid
# → embeddings/<tag>/detectors_tile.npz   (format detectors_final_v2.npz : mean/scale/coef/intercept)
# → embeddings/grid_scores_ensemble.npy   (moyenne des probas des seeds/mannequins)
```

`build_smart_queue.py` lit `grid_scores_v2.npy` + `detectors_final_v2.npz` en dur :
pointer ces deux noms sur les fichiers ci-dessus (ou adapter les 2 lignes) suffit à
faire tourner la file d'annotation sur l'ensemble SSL.

## 4. Protocole (pourquoi les chiffres sont comparables)

- **Folds appariés** : les plis spatiaux (`StratifiedGroupKFold`, blocs 10 m,
  `random_state=0`) sont calculés **une seule fois** sur les coordonnées et réutilisés
  pour tous les tags → chaque comparaison A/B est appariée.
- **Négatifs appariés** : fond aléatoire (rng seed 0, exclusion ±3 tuiles autour des
  annotations) et candidats rejetés (`candidates.sqlite`) ne dépendent pas du modèle →
  mêmes tuiles pour tous les tags (vérifié par assertion).
- **Alignement des annotations** vérifié au chargement (même nombre/ordre de points) —
  si `session.gpkg` a bougé, le script échoue au lieu de comparer des folds décalés.
- **Normalisation** : L2 puis `StandardScaler` (logreg/MLP), cosine (kNN) — identique à
  `analyze_bank.py` / `build_detectors.py`.
- **BLAS mono-thread** (AGENTS.md §4.8) ; parallélisation par processus uniquement.

## 5. Lire les résultats

| Observation | Interprétation |
|---|---|
| `ssl_seed*` > `base` en AUPRC (Δ ≥ +0.02 par espèce) | l'adaptation aérienne aide → garde le SSL dans l'outil |
| `ssl_seed*` ≈ `base` | l'adaptation n'apporte rien sur ce corpus (≈ le résultat du chapitre contexte Arctic, où entraîner ne battait pas le gelé aligné) |
| `ssl_seed*` < `base` | dérive : l'aérien coûte de la sémantique (hypothèse H3) |
| `ensemble` > `mean±std seeds` | l'ensemble réduit la variance des 3 seeds → à utiliser **dans l'outil**, pas au manuscrit |

⚠️ **AGENTS.md §4.4** : `mean±std` sur les seeds est l'estimateur canonique du mémoire.
Le **vote majoritaire / la moyenne de probas n'en est pas un** et ne doit pas être cité
comme F1 de modèle dans le manuscrit. Le JSON trace les deux champs séparément
(`canonical_mean_std_ssl_*` vs `ensemble_diagnostic`) : ne jamais les confondre.

## 6. Coûts mesurés / attendus

| Étape | Durée | Notes |
|---|---|---|
| `prepare_index.py` (local) | ~30 s | lecture d'en-têtes raster seulement |
| **Narval** `extract_web` par tag | ~1-2 min (GPU) | 6276 tuiles uniques, batch 256 |
| **Narval** bench logreg+kNN | ~1-2 min | 4 tags, 5 espèces |
| **Narval** 4 tags complets | **~10 min** | un seul job, dépendance `afterok` |
| local `bank` (raster, ctx 512) | ~15 min/modèle | CPU 16 threads |
| local `negatives` | ~20 min/modèle | idem |
| local `grid` (66 816 tuiles) | ~1,5-2 h/modèle | memmap fp16, repartable |
| bench logreg / kNN | < 1 min | — |
| bench MLP | ~10-15 min | `MLPClassifier` avec early stopping |

## 7. Fichiers livrés

| Fichier | Rôle |
|---|---|
| [`scripts/leo_ssl_annot_prepare_index.py`](leo_ssl_annot_prepare_index.py) | **local** : UTM → `idx` de tuile, produit `annot_index.npz` (155 ko) |
| [`scripts/leo_ssl_annot_extract_web.py`](leo_ssl_annot_extract_web.py) | **Narval GPU** : embeddings des annotations depuis les JPEG web, par modèle |
| [`scripts/slurm_leo_ssl_annot_eval.sh`](slurm_leo_ssl_annot_eval.sh) | **Narval** : un job = 4 tags + banc d'évaluation, `--dependency=afterok` |
| [`scripts/leo_ssl_annot_extract.sh`](leo_ssl_annot_extract.sh) | local : encode les annotations depuis le raster (contexte 512 inclus) |
| [`scripts/leo_ssl_annot_bench.py`](leo_ssl_annot_bench.py) | multiclass + détection + ensemble, folds appariés |
| `annotations_leo/extract_embeddings.py` | **modifié** : `--ckpt`, `--out-dir`, phase `negatives` (défauts inchangés) |
