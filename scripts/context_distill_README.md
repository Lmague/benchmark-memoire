# Self-distillation contexte→tuile (Design A) — DINOv3-B + LoRA, teacher DINOv3-L gelé

Branche : `exploration_nocturne`. Statut : **code + jobs SLURM prêts à soumettre, AUCUN
entraînement lancé** (contrainte de la mission — pas de GPU local, pas de lancement
Narval sans confirmation humaine). Tout chiffre cité ici vient d'un fichier du dépôt
lu ou d'un calcul exécuté localement (jamais inventé) ; les non-mesures sont marquées
explicitement.

## 1. Objectif

Un student DINOv3 ViT-B/16 LVD + LoRA apprend à encoder une **tuile 224px** en
s'aidant, **pendant l'entraînement seulement** (Design A), d'une **fenêtre de contexte
spatial** (512/1024/2048px, la même tuile mais vue plus large de son ortho-mère)
encodée par un **teacher DINOv3 ViT-L/16 LVD gelé** (stop-gradient). À l'inférence :
**un seul modèle, une seule entrée** (la tuile). But : mieux généraliser sur le
hold-out **spatial** (`splits_spatial/`), en particulier sur les classes confondues à
petite échelle (LICH/MOSS/SEDG).

Loss = `focal(classif, student CLS → logits 11cls) + λ · distill(proj(student CLS), teacher(contexte))`.

## 2. Fichiers livrés

| Fichier | Rôle |
|---|---|
| [context_crop.py](context_crop.py) | Découpe les fenêtres de contexte + reconstruit leurs coordonnées pixel (voir §4) |
| [context_distill.py](context_distill.py) | Boucle d'entraînement + extraction + sonde canonique |
| [../configs/context_distill_dinov3b.yaml](../configs/context_distill_dinov3b.yaml) | Config modèle/LoRA/optim/train (teacher et contexte restent des flags CLI) |
| [slurm_context_crop.sh](slurm_context_crop.sh) | Job Narval CPU (optionnel — voir §5) |
| [slurm_context_distill.sh](slurm_context_distill.sh) | Job Narval GPU (1× A100 40 Go) |

## 3. Ce qui est déjà établi (sourcé, à ne pas refaire)

- **Split d'évaluation** : `splits_spatial/frac100_seed{0,1,2}/{train,val,test}.csv` —
  hold-out par **orthomosaïque entière** (train/val/test disjoints par ortho). Vérifié :
  `train.csv` est **bit-identique** (md5) entre les 3 seeds à frac100 — pas de sous-
  échantillonnage stochastique à 100%, donc **un seul passage de `context_crop.py`
  couvre les 3 seeds**.
- **Schéma de labels des CSV spatiaux** : BRUT 12 classes (RHOL présent au train —
  152 lignes vérifiées, absent du val/test). `context_distill.py` filtre RHOL et
  remappe 12→11 **exactement** comme `scripts/datacurve_one_run.py` (import direct de
  sa table `LABEL_REMAP_12TO11`, pas de redéfinition).
- **LoRA r=2 (blocs 6-11)** comme point de départ : `results/lora_rank_ablation_CANONICAL.json`
  (r=2 → F1=0.4838±0.0020, meilleur que r=8 → 0.4829±0.0038) et
  `results/lora_block_ablation_CANONICAL.json` (blocs 6-11 → F1=0.4844±0.0011 vs blocs
  0-5 → 0.4790±0.0019). **⚠️ Ces deux écarts sont < l'écart-type inter-seed ≈0.008**
  (AGENTS.md §4.4, aucune différence <~0.01 n'est interprétable sans IC bootstrap) — un
  point de départ défendable, **pas un optimum démontré**, et ces ablations utilisaient
  le **split aléatoire**, pas le split spatial. À reconfirmer.
- **Baselines de comparaison** :
  - LoRA r=8 (12 blocs), split aléatoire, sans contexte : F1=0.4835±0.0011
    (`results/all_models_canonical_merged.json`, `results/lora_block_ablation_CANONICAL.json.baselines`).
  - **Courbe spatiale frac100 (LoRA r=8, 3 seeds)** — lu et calculé le 2026-08-29
    depuis `results/spatial_datacurve_CANONICAL.csv` (lignes `tag=frac100`) :
    seeds `[0.4885, 0.4807, 0.4789]` → **mean=0.4827, std=0.0042** (`n_train=49281`
    par seed). C'est le **comparateur pertinent pour R1/R2/R3** (même split spatial),
    **pas** le chiffre split-aléatoire 0.4835 ci-dessus. `n_train=49281` =
    49433 (lignes de `splits_spatial/frac100_seed0/train.csv`) − 152 (lignes RHOL) —
    cohérent avec le filtrage RHOL fait par `context_distill.py` (§3, schéma de
    labels).

## 4. Le problème des coordonnées pixel — résolu et VALIDÉ

`tile_XXXXX.png` n'encode pas sa position dans l'ortho : `scripts/tilerization.py`
incrémente `tile_count` pour chaque fenêtre glissante **non vide** (`tile_is_empty`),
mais ne sauve que celles qui intersectent une annotation — d'où les trous observés
(`tile_06871` → `tile_06894`). `context_crop.py` **réutilise directement** les
fonctions de `tilerization.py` (`tile_is_empty`, `get_dominant_class`, mêmes
constantes) pour rejouer exactement la même boucle et retrouver `(row_off, col_off)`.

**Validation empirique (2026-08-29)**, sur `20230724_alder39_m3m` (28369 fenêtres
testées, 2489 tuiles annotées) :
- Égalité d'ENSEMBLE parfaite entre les `tile_count` reconstruits et les
  `tile_XXXXX.png` réellement sur disque : **2489/2489**, 0 divergence de classe.
- Pixels **identiques bit-à-bit** (`np.array_equal` = True) entre le raster relu à la
  fenêtre reconstruite et le `tile_06868.png` déjà sur disque.
- Script de validation non livré (ponctuel, hors périmètre) mais reproductible :
  rejoue `process_one_raster` sans réécrire les PNG, compare aux fichiers existants.

**Condition de validité** : le replay est fidèle **si le raster n'a pas été réécrit**
depuis la tuilisation d'origine (`tile_is_empty` dépend des octets exacts). Si un COG a
changé, `context_crop.py` **échoue bruyamment** (assertion sur l'ensemble des
tile_count attendus + sur la classe dominante) plutôt que de produire un contexte
silencieusement désaligné.

`context_crop.py` a été testé de bout en bout (9 tuiles réelles, 3 classes, contexte
1024px, `python3 scripts/context_crop.py --split-csv ... --context-sizes 512,1024
--out-size 224`) — fonctionne, coordonnées vérifiées manuellement cohérentes avec le
centre de la tuile.

## 5. Décision de conception : où tourne `context_crop.py` ?

Les **38 COG bruts** (`High-Resolution Arctic Vegetation Maps.../photogrammetry/`)
ne sont présents QUE localement — comme tous les jobs Narval de ce dépôt, seul
`$SCRATCH/tiles.zip` (déjà tuilé 224px) est transféré, jamais les orthomosaïques
brutes (probablement des dizaines de Go).

**Chemin recommandé : exécuter `context_crop.py` EN LOCAL**, puis zipper
`context_<size>/` et le transférer vers `$SCRATCH/context_<size>.zip` — **exactement**
la convention `tiles.zip` déjà utilisée par tous les `slurm_*.sh` du dépôt :

```bash
python scripts/context_crop.py \
    --split-csv splits_spatial/frac100_seed0/train.csv \
    --context-sizes 512,1024,2048 --out-size 224 --out-dir out/context

cd out/context && for d in context_*; do zip -qr "../../${d}.zip" "$d"; done
scp ../../context_512.zip ../../context_1024.zip ../../context_2048.zip narval:$SCRATCH/
```

`scripts/slurm_context_crop.sh` est fourni comme **alternative CPU Narval**, mais
n'est utilisable QUE si tu transfères toi-même les COG bruts sous `$SCRATCH/` au
préalable (le script vérifie leur présence et échoue explicitement sinon).

## 6. Décision de conception : contexte redimensionné à 224px au stockage

Un contexte natif 2048px en PNG pèserait ~10 Mo/tuile → pour les 49433 tuiles train,
~500 Go. Intraitable à transférer/charger. `context_crop.py` **redimensionne toujours**
la fenêtre de contexte à `--out-size` (défaut 224, résolution native DINOv3) avant
sauvegarde — principe des "global crops" DINO/DINOv2 : le champ de vision réel varie
(512/1024/2048px dans l'espace pixel de l'ortho), mais le tenseur réseau reste de
taille constante. Conséquence directe :

- **Poids de stockage du même ordre de grandeur que `tiles.zip`** quel que soit
  `--context-size` — **mesuré sur le corpus complet** (49433 tuiles train, run réel
  du 2026-08-29, transféré sur Narval) :

  | Zip | Taille réelle |
  |---|---|
  | `context_512.zip` | 6,0 Go |
  | `context_1024.zip` | 6,1 Go |
  | `context_2048.zip` | **2,7 Go** |
  | **Total (les 3 tailles)** | **≈14,8 Go** |

  Contre-intuitif au premier abord : `context_2048.zip` est **plus léger** que
  512/1024, pas plus lourd. Explication : plus la fenêtre source est grande, plus le
  redimensionnement LANCZOS vers 224px moyenne (lisse) l'image — un contexte 2048px
  condense ~9× plus de pixels sources dans le même 224×224 qu'un contexte 512px
  (~2,3×), donc moins de détail haute fréquence survit et le PNG résultant se
  compresse mieux. Pas un bug ; confirmé par une inspection visuelle du principe
  (LANCZOS = filtre passe-bas avant sous-échantillonnage). Estimation initiale
  (§ précédente version de ce document, extrapolée sur 9 échantillons à 1024px
  seulement) : ~6,2 Go/taille — correcte pour 512/1024, sous-estimait l'écart avec
  2048px faute d'avoir mesuré cette taille avant le run complet.
- **Coût mémoire/calcul du teacher identique entre R1/R2/R3** (toujours une image
  224×224 en entrée) — d'où `--mem=64G` uniforme dans `slurm_context_distill.sh`, PAS
  de bump à 128G pour 2048px (divergence assumée par rapport à l'énoncé initial, qui
  anticipait un contexte à résolution native).
- Le "token d'échelle" (§8) devient le SEUL vecteur d'information sur le GSD effectif
  de la vue (puisque le tenseur lui-même ne le révèle plus directement).

## 7. Contexte nécessaire pour le TRAIN uniquement

Design A : à l'inférence (et pour l'évaluation val/test, y compris la métrique
d'early-stopping et la sonde canonique finale), le student ne voit QUE la tuile
224px. `context_crop.py` n'a donc besoin de traiter que `train.csv` — pas val/test.
`--split-csv` accepte plusieurs CSV si un usage futur en avait besoin, mais l'usage
recommandé (§5, §9) ne traite que le train.

## 8. Token d'échelle (GSD) — choix d'implémentation explicite

Le student ne voyant QUE la tuile 224px (Design A), sa vue physique est **constante**
pendant tout l'entraînement — un token d'échelle n'a donc de sens que pour la branche
de **projection** (l'alignement élève→enseignant), pas pour la classification. Implémenté
comme un petit MLP sur `log(context_size / 224)`, dont la sortie est **concaténée** à
la feature CLS student avant la tête de projection `768(+32)→1024`. **Ce n'est PAS un
token de séquence transformer inséré dans DINOv3-HF** — `AutoModel.forward` ne
l'expose pas sans monkey-patcher les embeddings internes (jugé trop risqué pour un
chemin de code partagé par tout le pipeline canonique). Choix documenté explicitement
pour ne pas faire passer une décision d'implémentation pour une exigence de l'énoncé.

## 9. Comment lancer (ordre R1 → R2/R3)

### Étape 0 — préparer le contexte (une fois, en local, §5)

```bash
python scripts/context_crop.py \
    --split-csv splits_spatial/frac100_seed0/train.csv \
    --context-sizes 512,1024,2048 --out-size 224 --out-dir out/context
cd out/context && for d in context_*; do zip -qr "../../${d}.zip" "$d"; done
scp ../../context_{512,1024,2048}.zip narval:$SCRATCH/
```

### Étape 1 — R1 (contexte 1024px, principal), 3 seeds en parallèle

```bash
sbatch scripts/slurm_context_distill.sh 0        # context_size=1024 par défaut
sbatch scripts/slurm_context_distill.sh 1
sbatch scripts/slurm_context_distill.sh 2
```

Résultats : `$SCRATCH/context_distill/runs/dinov3_vitb16_lvd_ctxdistill_ctx1024_r2a4_frac100_seed{0,1,2}/metrics.json`.

### Étape 2 — SI R1 bat la baseline spatiale (frac100, LoRA r=8 : F1=0.4827±0.0042,
`results/spatial_datacurve_CANONICAL.csv`, cf. §3) → R2 (512px) et R3 (2048px)

```bash
sbatch scripts/slurm_context_distill.sh 0 512
sbatch scripts/slurm_context_distill.sh 1 512
sbatch scripts/slurm_context_distill.sh 2 512
sbatch scripts/slurm_context_distill.sh 0 2048
sbatch scripts/slurm_context_distill.sh 1 2048
sbatch scripts/slurm_context_distill.sh 2 2048
```

Puis tracer la courbe 512→1024→2048 (`f1_macro_pres_test`, `balanced_accuracy_test`
depuis chaque `metrics.json`).

### Hyperparamètres par défaut (`configs/context_distill_dinov3b.yaml`)

| Clé | Valeur | Source |
|---|---|---|
| LoRA r / alpha / blocs | 2 / 4.0 / [6..11] | §3 (point de départ, pas un optimum) |
| λ (distillation) | 1.0 | flag `--lambda-distill`, mission |
| focal γ | 2.0 | flag `--focal-gamma`, mission |
| lr lora/head/proj | 1e-4 | convention LoRA existante du dépôt |
| epochs / patience | 50 / 10 | convention existante |
| batch_size | 64 | marge A100 40 Go avec teacher ViT-L additionnel (non mesuré) |
| early_stop_metric | `f1_macro_pres` | convention du dépôt ; `balanced_accuracy` calculée et journalisée à chaque epoch mais pas utilisée pour la sélection |

## 10. Non-fuite spatiale

`splits_spatial/` sépare train/val/test par **ortho entière**. Une fenêtre de contexte
est découpée dans le **même raster** que sa tuile (jamais un autre), donc un contexte
de train ne peut, par construction, jamais piocher des pixels d'un ortho val/test —
aucune fuite additionnelle introduite par `context_crop.py`.

## 11. Politique de bord

Une fenêtre 2048px centrée sur une tuile proche du bord d'un ortho déborderait
souvent. Politique : **clamp** (la fenêtre glisse pour rester entièrement dans le
raster, n'est alors plus rigoureusement centrée) plutôt que padding noir (qui
introduirait un signal artificiel — `tile_is_empty` traite justement le noir comme
"absence de donnée" ailleurs dans le pipeline). Chaque fenêtre réellement lue est
enregistrée dans `coords_manifest.json` (`row0`/`col0`/`clamped`) — pas de fenêtre
devinée côté entraînement.

## 12. Réutilisation du code existant (pas de réimplémentation)

- `src.models.build_model(..., regime="lora", lora=cfg.lora)` — injection LoRA (student).
- `src.models.build_frozen_extractor` — chargement teacher + n'importe quel backbone frozen.
- `src.models.load_finetuned_ssl_backbone` + `merge_lora_state_dict` — extraction
  post-entraînement (Design A, tuile seule) : le checkpoint sauvé par
  `context_distill.py` (`model_state_dict` = état du classifieur LoRA `backbone.*`/`head.*`
  **pur**, sans mélange avec `proj`/`scale_mlp` — ceux-ci sont retournés séparément par
  `build_student` précisément pour garantir cette pureté) est **directement**
  rechargeable par cette fonction, sans adaptation.
- `src.engine.build_optimizer` / `build_scheduler` — AdamW multi-groupes + cosine step-based.
- `src.losses.build_class_weights` — pondération 1/√n (même schéma que tout le dépôt).
- `src.utils.make_canonical_lr` — sonde linéaire canonique (lbfgs, mono-thread).
- `scripts.datacurve_one_run._extract_backbone_embeddings` / `_apply_11cls_remap` —
  extraction + remap 12→11, **même code** que le pipeline LoRA canonique.
- `scripts.tilerization.tile_is_empty` / `get_dominant_class` — réutilisées par
  `context_crop.py` pour ne jamais diverger de la logique de tuilisation originale.

## 13. Validation effectuée (CPU, sans GPU) avant livraison

- Replay de coordonnées : voir §4 (égalité d'ensemble + pixels identiques).
- `context_crop.py` : run réel de bout en bout (9 tuiles, 3 classes, contexte
  512+1024px) — sorties correctes, coordonnées cohérentes.
- `context_distill.py` : run réel de bout en bout SUR CPU (1 epoch, 9 tuiles, batch=2)
  avec les vrais poids DINOv3-B/DINOv3-L (cache HF local) — boucle d'entraînement
  (focal + distillation, backward, optimizer step), sélection du best checkpoint,
  extraction via `load_finetuned_ssl_backbone` (**211/211 clés chargées, 0 manquante**
  — confirme que le merge LoRA + la convention de checkpoint sont corrects), sonde
  canonique : **tout s'exécute sans erreur**. Métriques du smoke test NON
  significatives (9 échantillons, train=val=test) — sert uniquement à valider
  l'absence de bug, pas la performance du modèle.
- **`context_crop.py` sur le corpus complet** : exécuté le 2026-08-29 (49433 tuiles
  train, 15 orthos, `--context-sizes 512,1024,2048`), zippé et transféré sur
  `$SCRATCH` (`/scratch/lmague/`) — tailles réelles en §6. Durée d'exécution locale
  non chronométrée précisément par cette session (lancée par l'utilisateur dans son
  propre terminal).
- **Non fait** (hors périmètre "pas de GPU") : entraînement réel sur Narval, mesure du
  temps GPU réel, validation du budget mémoire 64G sous charge réelle.

## 14. `--design B`

Non implémenté (`NotImplementedError` explicite dans `context_distill.py`) — la
mission demande le flag mais ne spécifie que Design A. Deviner l'architecture de
Design B (contexte à l'inférence ? deux modèles ?) aurait été plus risqué qu'un gap
honnête.
