# Self-distillation contexte→tuile — DINOv3-B + LoRA

Branche : `exploration_nocturne`. Statut : **code + jobs SLURM prêts à soumettre, AUCUN
entraînement lancé** (contrainte de la mission — pas de GPU local, pas de lancement
Narval sans confirmation humaine). Tout chiffre cité ici vient d'un fichier du dépôt
lu ou d'un calcul exécuté localement (jamais inventé) ; les non-mesures sont marquées
explicitement.

## 1. Objectif

Un student DINOv3 ViT-B/16 LVD + LoRA apprend à encoder une **tuile 224px** en
s'aidant d'une **fenêtre de contexte spatial** (512/1024/2048px, la même tuile mais
vue plus large de son ortho-mère). But : mieux généraliser sur le hold-out **spatial**
(`spatial_datacurve/splits/`), en particulier sur les classes confondues à petite échelle
(LICH/MOSS/SEDG). Loss (toujours) = `focal(classif) + λ · distill(proj(feat_tuile), teacher(contexte))`.

Trois axes de variation, indépendants (`context_distill.py --design {A,B} --teacher {externe|ema_self} --context-size {512,1024,2048}`).

**⚠️ Convention de nommage (pour éviter la confusion vécue le 2026-08-29)** : R1/R2/R3
désignent ici les **3 lignes du tableau de l'utilisateur** ("Le plan final"), PAS des
tailles de contexte. Les variantes de taille de contexte (512/1024/2048px, toutes en
Design A + teacher DINOv3-L) sont désignées SANS numéro — "contexte 512px" /
"contexte 1024px" / "contexte 2048px" — précisément pour ne jamais entrer en
collision avec R1/R2/R3 :

| Run | Teacher | Design | Question posée |
|---|---|---|---|
| **R1 (principal)** | DINOv3-L/16 LVD gelé, externe, contexte **1024px** | A (contexte au train seulement, 1 tuile à l'inférence) | Self-distillation contexte→tuile de base |
| **R2** | idem R1 (DINOv3-L, contexte 1024px, INCHANGÉ) | **B** (tuile+contexte à l'inférence) | Le contexte aide-t-il EN PLUS au moment de prédire (pas seulement à l'entraînement) ? |
| **R3** | copie **EMA** du student lui-même (`ema_self`), contexte 1024px | A | Le gain vient-il d'un teacher externe plus riche, ou juste d'un signal de self-distillation ? |
| *(extension, hors tableau)* contexte 512px | idem R1 | A | Effet de la taille de contexte (plus petite) |
| *(extension, hors tableau)* contexte 2048px | idem R1 | A | Effet de la taille de contexte (plus grande) |

Design A : à l'inférence, un seul modèle, une seule entrée (la tuile). Design B (R2) :
même modèle mais deux entrées (tuile+contexte) — "borne supérieure théorique", pas
forcément un design déployable, cf. §14.

## 2. Fichiers livrés

| Fichier | Rôle |
|---|---|
| [context_crop.py](context_crop.py) | Découpe les fenêtres de contexte + reconstruit leurs coordonnées pixel (voir §4) |
| [context_distill.py](context_distill.py) | Boucle d'entraînement + extraction + sonde canonique |
| [../configs/context_distill_dinov3b.yaml](../configs/context_distill_dinov3b.yaml) | Config modèle/LoRA/optim/train (teacher et contexte restent des flags CLI) |
| [slurm_context_crop.sh](slurm_context_crop.sh) | Job Narval CPU (optionnel — voir §5) |
| [slurm_context_distill.sh](slurm_context_distill.sh) | Job Narval GPU (1 job = 3 seeds séquentiels, slice MIG `a100_3g.20gb`) |

## 3. Ce qui est déjà établi (sourcé, à ne pas refaire)

- **Split d'évaluation** : `spatial_datacurve/splits/frac100_seed{0,1,2}/{train,val,test}.csv` —
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
    par seed). C'est le **comparateur pertinent pour R1 et ses variantes de contexte
    (512/1024/2048px)** (même split spatial),
    **pas** le chiffre split-aléatoire 0.4835 ci-dessus. `n_train=49281` =
    49433 (lignes de `spatial_datacurve/splits/frac100_seed0/train.csv`) − 152 (lignes RHOL) —
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
    --split-csv spatial_datacurve/splits/frac100_seed0/train.csv \
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
- **Coût mémoire/calcul du teacher identique entre les tailles de contexte** (toujours une image
  224×224 en entrée) — d'où `--mem` uniforme dans `slurm_context_distill.sh` (RAM
  système, pas VRAM) quel que soit `--context-size`, PAS de bump à 128G pour 2048px
  (divergence assumée par rapport à l'énoncé initial, qui anticipait un contexte à
  résolution native).
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

## 9. Comment lancer

### Étape 0 — préparer le contexte (une fois, en local, §5)

```bash
python scripts/context_crop.py \
    --split-csv spatial_datacurve/splits/frac100_seed0/train.csv \
    --context-sizes 512,1024,2048 --out-size 224 --out-dir out/context
cd out/context && for d in context_*; do zip -qr "../../${d}.zip" "$d"; done
scp ../../context_{512,1024,2048}.zip narval:$SCRATCH/
```

### Étape 1 — R1 (contexte 1024px, principal), 1 job, 3 seeds séquentiels

```bash
sbatch scripts/slurm_context_distill.sh        # context_size=1024, design=A, teacher=dinov3_vitl16_lvd (défauts)
```

Un seul job, une seule allocation GPU (slice MIG `a100_3g.20gb`, cf. §16 — pas un
A100 complet) : les tuiles+contexte sont extraits une fois, les 3 seeds tournent
l'un après l'autre dessus. Résultats :
`$SCRATCH/context_distill/runs/dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100_seed{0,1,2}/metrics.json`.

### R2 (Design B) et R3 (EMA self-teacher) — cf. §14-15

R2 nécessite un prérequis (contexte val/test, cf. §14) ; R3 aucun. Les deux peuvent
tourner **en même temps que R1**, ce sont des jobs indépendants — pas d'ordre
imposé (2026-08-29 : abandon d'une gate "attendre R1" suggérée initialement par
Claude, l'utilisateur préfère tout lancer ensemble) :

```bash
sbatch scripts/slurm_context_distill.sh 1024 A ema_self   # R3, aucun prérequis
sbatch scripts/slurm_context_distill.sh 1024 B             # R2, cf. §14 pour le prérequis
```

### Extension optionnelle (hors tableau) — effet de la taille de contexte

Pas dans le tableau de l'utilisateur, mais construit dès le départ (mission
initiale) : R1 avec un contexte plus petit (512px) ou plus grand (2048px), toujours
teacher DINOv3-L + Design A. À lancer si la courbe 512→1024→2048 intéresse, sans
gate particulière non plus :

```bash
sbatch scripts/slurm_context_distill.sh 512    # 1 job, 3 seeds
sbatch scripts/slurm_context_distill.sh 2048   # 1 job, 3 seeds
```

Puis tracer la courbe 512→1024→2048 (`f1_macro_pres_test`, `balanced_accuracy_test`
depuis chaque `metrics.json`).

Chaque `sbatch` ci-dessus = **1 job = 3 seeds séquentiels** (pas d'argument seed —
retiré du script, cf. §16). 5 jobs au total pour couvrir R1, R2, R3 et les 2
variantes de contexte (15 runs), au lieu de 15 jobs séparés.

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

`spatial_datacurve/splits/` sépare train/val/test par **ortho entière**. Une fenêtre de contexte
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
  post-entraînement : le checkpoint sauvé par `context_distill.py`
  (`model_state_dict` = état du classifieur LoRA `backbone.*`/`head.*` **pur**, sans
  mélange avec `proj`/`scale_mlp` — ceux-ci sont construits/retournés séparément
  précisément pour garantir cette pureté, cf. `build_projection_head` dans le script)
  est **directement** rechargeable par cette fonction, sans adaptation. Vrai sous
  Design A (tête 768→11) ET Design B (tête 1536→11) : `load_finetuned_ssl_backbone`
  ignore de toute façon `head.*` en reconstruisant le backbone nu pour la sonde.
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
- **EMA self-teacher** (`--teacher ema_self`) : run CPU réel de bout en bout (mêmes 9
  tuiles, vrais poids DINOv3-B) — un seul modèle chargé (211/211 clés, confirmant
  qu'aucun teacher externe n'est téléchargé), boucle EMA + entraînement + extraction +
  sonde exécutés sans erreur. `n_train` diffère correctement de R1 (700 011 params
  entraînables vs 905 067 — delta exact = taille de la tête de projection
  800×(1024−768)+(1024−768), cohérent avec teacher_dim=768 au lieu de 1024).
- **Design B** : contexte val/test généré pour la fixture de 9 tuiles, run CPU réel de
  bout en bout — tag `dB_tL` distinct de R1 (`dA_tL`), tête 1536→11 correctement
  utilisée (913 515 params entraînables, delta exact = 768×11 vs la tête 768→11 de A),
  extraction fusionnée confirmée `(9, 1536)` sur train/val/test, sonde exécutée sans
  erreur.
- **Non fait** (hors périmètre "pas de GPU") : entraînement réel sur Narval, mesure du
  temps GPU réel, validation du slice MIG 20 Go sous charge réelle (cf. §16), mesure
  du surcoût réel du 3e forward (student sur contexte) sous Design B.

## 14. R2 — Design B (`--design B`)

Ajouté le 2026-08-29. **Contrainte de conception (validée avant implémentation,
cf. discussion du 2026-08-29) : Design B garde l'architecture de R1 STRICTEMENT
INTACTE** (teacher DINOv3-L externe, distillation, λ, focal — rien ne change) et
modifie **UNE SEULE chose** : la tête de classification prend
`[feat_tuile ; feat_contexte]` (2×768) au lieu de `feat_tuile` seule, les deux
features venant du **même backbone LoRA student** (poids partagés, deux forwards
par batch). La branche de distillation reste identique à R1 (seule `feat_tuile` est
projetée et comparée au teacher).

Pourquoi ce choix plutôt qu'une architecture plus simple (fusion pure, sans
teacher/distillation) : la question posée par Design B est *"le contexte aide-t-il
EN PLUS à l'inférence"* — cette question n'a de sens que si R1 et R2 ne diffèrent
que par CETTE variable. Simplifier l'architecture (supprimer le teacher) aurait
fait varier trois choses à la fois (entrée d'inférence, présence du teacher,
objectif d'entraînement), rendant un éventuel gain ininterprétable.

**Prérequis supplémentaire (contrairement à R1, R3, et aux variantes de contexte
512/2048px) : le contexte doit exister pour VAL ET TEST aussi, pas seulement
train** — puisque R2 en a besoin à l'évaluation (early-stopping par epoch) et à
l'extraction finale, pas seulement à l'entraînement :

```bash
python scripts/context_crop.py \
    --split-csv spatial_datacurve/splits/frac100_seed0/val.csv spatial_datacurve/splits/frac100_seed0/test.csv \
    --context-sizes 1024 --out-size 224 --out-dir out/context   # MÊME --out-dir que le run train
cd out/context && zip -qr ../../context_1024.zip context_1024   # ré-zippe avec val/test inclus
scp ../../context_1024.zip narval:$SCRATCH/                     # écrase l'ancien (train seul)
```

Puis :

```bash
sbatch scripts/slurm_context_distill.sh 1024 B
```

`slurm_context_distill.sh` vérifie la présence du contexte val avant de lancer
l'entraînement (échec explicite sinon, plutôt qu'un plantage tardif et confus).

Extraction/sonde : `context_distill.py` utilise une fonction dédiée
(`_extract_fused_embeddings`, PAS `_extract_backbone_embeddings` réutilisée par
Design A) qui recharge le backbone (même mécanisme `load_finetuned_ssl_backbone`)
puis forward tuile+contexte pour train/val/test, concatène (N, 1536), et passe ça à
la MÊME sonde canonique (`_run_probe_with_balanced_acc`, agnostique à la
dimensionnalité des features).

## 15. R3 — EMA self-teacher (`--teacher ema_self`)

Ajouté le 2026-08-29 à la demande de l'utilisateur (isoler la source du gain de R1 :
teacher externe plus riche, ou juste signal de self-distillation). Teacher = copie
**EMA (momentum, `--ema-momentum`, défaut 0.999)** des paramètres **entraînables**
du backbone student (LoRA A/B + LayerNorm) — PAS de modèle externe chargé. La base
DINOv3 gelée est identique par construction entre les deux copies (jamais mise à
jour côté student), donc l'EMA dessus serait un no-op — exclue du calcul pour ne pas
itérer sur ~86M paramètres inutilement à chaque step.

Construit dans `context_distill.py` (`build_ema_teacher`) APRÈS avoir placé le
student sur le device (`classifier.to(device)`), pour que la copie EMA et
l'appariement (teacher_param, student_param) référencent directement les tenseurs
CUDA finaux — évite tout risque lié au comportement de `Module.to()` sur l'identité
des `Parameter` selon la version de PyTorch. `teacher_dim` devient 768 (embed_dim du
student) au lieu de 1024 (DINOv3-L) — la tête de projection s'adapte automatiquement
via `build_projection_head(embed_dim, teacher_dim)`.

Contexte : **1024px, comme R1** (confirmé par l'utilisateur) — isole une seule
variable (le teacher) par rapport à R1, comparaison la plus propre.

```bash
sbatch scripts/slurm_context_distill.sh 1024 A ema_self
```

Aucun prérequis supplémentaire (même contexte que R1, aucun teacher externe à
charger — en fait UN téléchargement HF de moins que R1).

## 16. Structure des jobs (2026-08-29, révisé à la demande de l'utilisateur)

`slurm_context_distill.sh` a changé deux fois pendant cette expérience — les deux
choix sont documentés ici pour éviter la confusion si une ancienne version traîne
quelque part.

**Job = 1 config × 3 seeds séquentiels** (version actuelle), PAS 3 jobs séparés (1
par seed, version précédente) : l'utilisateur n'est pas pressé et préfère minimiser
le nombre d'allocations GPU plutôt que paralléliser pour un temps d'horloge plus
court. Une seule extraction de tuiles+contexte par job (au lieu de 3), les 3 seeds
réutilisent la même allocation — convention déjà utilisée ailleurs dans le dépôt
(`slurm_lora_rank_ablation.sh`, `slurm_datacurve_spatial.sh`). Plus aucun argument
`seed` en CLI : `$1`/`$2`/`$3` = context_size/design/teacher.

**Slice MIG `a100_3g.20gb` (20 Go), PAS un A100 complet** — GPU minimal jugé
suffisant, sur demande explicite de l'utilisateur ("le moins de GPU possible").
Raisonnement (documenté en tête de `slurm_context_distill.sh`, PAS mesuré sur ce
script précis) : le teacher (externe ou EMA) ne fait qu'un FORWARD (jamais de
backward, jamais d'état optimizer) — nettement moins coûteux qu'un entraînement
complet. Le précédent le plus proche mesuré dans ce dépôt (`slurm_lora_dinov3_vitl.sh`)
ENTRAÎNE (forward+backward+optimizer LoRA) un DINOv3-L **complet** dans un slice
`a100_3g.20gb` — mon student est plus petit (DINOv3-B) et le teacher n'ajoute qu'un
forward, donc a priori plus léger. Si le job OOM malgré tout : replier sur
`--gres=gpu:a100:1` + `--mem=60G` (A100 complet, 40 Go, queue plus lente — commenté
dans le script). **À vérifier sur le premier run réel** — c'est un jugement raisonné
par analogie, pas une mesure.

## 17. Incident (2026-08-30) : `splits_spatial/` n'a jamais existé sur Narval

Le premier run réel de R1 (job 2073957) s'est terminé "avec succès" (exit 0) mais
**sans rien faire** — les 3 seeds ont été sautés avec
`[ERROR] split spatial absent: /home/lmague/benchmark-memoire/splits_spatial/frac100_seed{0,1,2}/train.csv`.

**Cause racine** : `splits_spatial/` (le répertoire utilisé PARTOUT dans ce code
depuis le début, hérité tel quel du prompt de mission initial) n'est **pas
whitelisté** dans `.gitignore` (`git check-ignore -v` confirme, catch-all `/*`) —
il n'a donc **jamais été commité**, jamais poussé sur GitHub, et n'existe pas dans
le clone `$HOME/benchmark-memoire` sur Narval. Le job n'a pas planté (le `for SEED`
continue sur erreur, cf. `slurm_context_distill.sh`) — il a juste tourné à vide
pendant tout le temps alloué. **Aucune alerte visible dans le `.out`**, seulement
dans le `.err` — à vérifier systématiquement en plus du `.out`.

**Ce qui existe RÉELLEMENT sur Narval** : `spatial_datacurve/splits/` — un répertoire
**différent**, généré par `spatial_datacurve/make_spatial_datacurve.py` ("v2",
2026-08-13, un jour après `splits_spatial/` du 2026-08-12), whitelisté dans
`.gitignore` (`!/spatial_datacurve/`) et donc bien présent sur Narval. C'est aussi
la source de `results/spatial_datacurve_CANONICAL.csv` (le comparateur utilisé
partout dans ce document, §3) — via `scripts/rapport/probe_spatial_canonical.py`,
qui re-probe les embeddings des runs `lora_spatial_v2/` entraînés sur ce split.

**Vérifié avant de corriger (2026-08-30)** : `spatial_datacurve/splits/frac100_seed{0,1,2}/
{train,val,test}.csv` et `splits_spatial/frac100_seed{0,1,2}/{train,val,test}.csv`
contiennent **exactement les mêmes lignes** (diff sur fichiers triés = 0 différence,
pour les 3 seeds) — seul l'ORDRE des lignes diffère (d'où des md5 différents sur les
fichiers non triés). Même ortho split train/val/test (15/8/9), même absence de
recouvrement d'ortho train↔val/test dans les deux. **Aucune donnée n'est en cause,
aucun résultat déjà cité dans ce document n'est invalidé** — seul le CHEMIN était
faux. Tous les scripts (`context_crop.py`, `context_distill.py`, `slurm_*.sh`,
ce README) ont été corrigés pour pointer vers `spatial_datacurve/splits/`. Les
crops de contexte déjà produits localement (`out/context/context_*`) restent
valides tels quels (mêmes tuiles, mêmes chemins relatifs).

**Pourquoi ça n'a pas été détecté avant** : jamais testé sur Narval avant ce run
(contrainte de la mission — pas de GPU local, donc pas de moyen de découvrir cette
absence plus tôt que le premier vrai `sbatch`).

## 18. Expérience 2026-09 — student SimDINOv2-B, contexte 512px, Design B

Ajouté 2026-09. Objectif : croiser les DEUX meilleures conclusions du sweep multi-backbone
frozen — (a) **SimDINOv2-B@512 est le pic frozen** (fused 0.5059), (b) la fusion **Design B
entraînée** (tête apprise [tuile;contexte], qui a porté DINOv3-B à 0.508) dépasse la sonde
linéaire frozen — en AFFINANT un student **SimDINOv2-B** avec la distillation contexte→tuile
+ tête fusionnée.

**Config : `configs/context_distill_simdinov2b.yaml`** — student SimDINOv2-B (iNat21
Plantae) + LoRA r=2 α=4 blocs 6-11 (mêmes hyperparams que le DINOv3-B de R1/R2).
Deux clés NOUVELLES résolues contre `cfg.paths.ckpt_dir` (Narval = `${SCRATCH}/checkpoints`) :
`checkpoint: simdinov2_vitb_inat21plantae.pth` (student) et
`teacher_checkpoint: simdinov2_vitl_inat21plantae.pth` (teacher SimL).

**Teacher SimDINOv2-L, PAS DINOv3-L — deux raisons :**
1. **Norme (bloquant)** : la fenêtre de contexte est normalisée au TRAIN avec la norme du
   teacher mais à L'ÉVAL avec celle du student (`_make_train_loader`/`_make_eval_loader_with_context`).
   SimB+SimL → les deux `simdino_inat` (pas de skew) ; SimB+DINOv3-L → `imagenet`≠`simdino_inat`
   → **garde-fou explicite dans `context_distill.py`** : Design B lève une `ValueError` si
   `teacher_norm_key != cfg.model.norm` (DINOv3-L acceptable seulement sous Design A, où le
   contexte n'est utilisé qu'au train).
2. **Même famille** : SimL est le meilleur extracteur de contexte SEUL du sweep frozen
   (ctx-seul 0.4941 @512) et partage le pré-entraînement plantae de SimB.

**Lancement (Design B @512, 3 seeds séquentiels, 1 job MIG a100_3g.20gb) :**

```bash
sbatch scripts/slurm_context_distill.sh 512 B simdinov2_vitl16 configs/context_distill_simdinov2b.yaml
```

`slurm_context_distill.sh` accepte maintenant **$4 = config** (défaut : la config DINOv3-B),
et sous Design B fusionne `context_512.zip` (train, ~49433 crops) **+**
`context_512_valtest.zip` (val=13209 + test=17598) dans le même `$CONTEXT_DIR` — même
convention que `slurm_context_frozen_models.sh`. Le `.pth` teacher SimL (1,24 Go) est déjà
dans `$SCRATCH/checkpoints/` (utilisé par le sweep frozen).

Résultats attendus : `$SCRATCH/context_distill/runs/
simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r2a4_frac100_seed{0,1,2}/metrics.json`.

### 18bis. Soutien littéraire — taille de contexte aérien (papier local)

Gomes et al., *Efficient Spatiotemporal Vegetation Pixel Classification With Vision
Transformers*, IEEE JSTARS vol. 19, 2026, DOI 10.1109/JSTARS.2026.3694818 — PDF dans
`Papiers/Efficient Spatiotemporal Vegetation PixelClassification With Vision Transformers.pdf`.
Leur §VI-B (scalability) sur **Serra do Cipó (imagerie aérienne UAV)** trouve le MÊME
comportement que notre sweep 512→1024→2048 :

> "for the Serra do Cipó dataset (aerial UAV imagery), increasing the square context window
> beyond a certain threshold proved detrimental … [optimal] peaked at 13×13 … As the window
> expanded, accuracy degraded … excessively large square context windows may inadvertently
> encompass multiple canopies … which may lead to less discriminative features."

TÂCHE aérien → petit contexte optimal ; TÂCHE au sol (Itirapina, surface proche) → grand
contexte optimal. Notre Arctic-TVC est du drone VHR (~1,5 mm/px) → se range côté Serra do
Cipó : **512px = meilleur contexte ; 2048px dégrade la discriminabilité** (appui indépendant
à la courbe 512>1024≫2048 observée sur les 5 backbones frozen et à la décision de n'affiner
que @512).

## 18ter. Résultats SimB@512 entraîné (Design B) + variante SANS teacher (λ=0)

Premier run réel (job 2549240, 2026-09, 3 seeds séquentiels, teacher SimL, λ=1.0) :

| seed | f1_macro_pres_test | best_C |
|---|---|---|
| 0 | 0.5028 | 0.001 |
| 1 | 0.5043 | 0.001 |
| 2 | (à compléter) | |

**Lecture** : le fine-tuning + distillation **n'a pas battu le frozen SimB@512 (0.5059, seed0,
sans entraînement)** — moyenne 2 seeds ≈ 0.5036, écart < σ inter-seed (≈0.008, AGENTS.md §4.4).
Constat cohérent avec le §18 et la question « le teacher est-il utile ? » :
- **En Design B, la distillation est INERTE** : la loss distill plafonne (~0.12-0.13) sans
  aider la classification (val F1 stagne → early stop 15-20/50 epochs). Le teacher SimL a
  coûté un forward ViT-L gelé par batch pour zéro gain.
- **SimDINOv2-B frozen est déjà au plafond** (0.5059) — beaucoup moins de marge que
  DINOv3-B (frozen 0.4953 → R2 entraîné 0.508, +0.013). Résultat négatif UTILE pour le
  manuscrit : « contexte + iNat frozen ≈ meilleure fusion entraînée, à coût GPU nul ».

**Conséquence pipeline** (patché 2026-09) : `--lambda-distill ≤ 0` désactive **vraiment** le
teacher — plus de chargement du .pth, plus de forward de distillation, plus de tête proj
(`teacher=None`, `proj=None`) ; le contexte est alors normalisé avec la norme student au
train comme à l'éval (cohérent par construction). `slurm_context_distill.sh` accepte
**$5=λ** et **$6=lora_rank** (alpha=2r automatique) :

```bash
# SANS teacher — prouver que λ=0 ≥ λ=1 en Design B (le vrai test de l'utilité du teacher)
sbatch scripts/slurm_context_distill.sh 512 B simdinov2_vitl16 \
       configs/context_distill_simdinov2b.yaml 0

# Avec LoRA r=8 au lieu de r=2 (r=2 vient de l'ablation DINOv3 ; SimDINOv2 pourrait
# avoir besoin de plus de capacité — cf. configs/simdinov2_vitb16_lora.yaml, r=8)
sbatch scripts/slurm_context_distill.sh 512 B simdinov2_vitl16 \
       configs/context_distill_simdinov2b.yaml 1.0 8
```
