# Dose-réponse d'alignement — SSL aérien Léo (ExPLoRA) → Arctic-TVC

Branche : dépôt Mémoire. Statut : **code livré, aucun calcul lancé** (les checkpoints
SSL Léo ne sont pas encore rapatriés). Question posée : *comment l'alignement du
pré-entraînement agit-il ?*

## 1. Pourquoi cette expérience

Les comparaisons existantes de pré-entraînement (`docs/report/sections/pretrain.tex`,
`configs/compare_simdinov2_pretraining.yaml`) **confondent** architecture, corpus et
objectif : on ne peut pas attribuer leur écart à « l'alignement ». Le SSL ExPLoRA-Léo
est la seule **manipulation contrôlée** du dépôt :

| | base | manipulation |
|---|---|---|
| architecture | SimDINOv2-B/16 | identique |
| objectif | DINO + iBOT + KoLeo | identique |
| corpus de départ | iNat21 Plantae | identique |
| **domaine** | photos de plantes au sol | **SSL continué sur orthos aériennes Léo** |

Trois hypothèses concurrentes, départagées par la trajectoire :

- **H1 — alignement d'apparence** : l'aérien suffit → le F1 gelé monte avec la dose ;
- **H2 — alignement taxonomique** : iNat Plantae porte tout → le F1 gelé reste plat ;
- **H3 — dérive** : l'aérien éloigne de la sémantique plante → le F1 gelé baisse.

## 2. Les trois compteurs mesurés (aucun entraînement supervisé)

Sonde linéaire **canonique** (`context_distill._run_probe_with_balanced_acc` :
StandardScaler float32 + `make_canonical_lr` lbfgs multinomial, grille
C ∈ {1e-4…10}, sélection sur val `f1_macro_pres`, refit, BLAS mono-thread) sur les
features **frozen** de chaque checkpoint, contexte 512px :

| Compteur | dim | Référence SimB-iNat @512 (à battre) | Lecture |
|---|---|---|---|
| `tile` | 768 | 0.4717 | transférabilité de la tuile seule |
| `ctx` | 768 | **0.4931** | décodabilité du voisinage — le mètre qui sépare le mieux les backbones (DINOv3-LVD : 0.4592) |
| `fused` | 1536 | **0.5059** | référence gelée du chapitre contexte |

Le point **époque 0** de la trajectoire est le backbone SimDINOv2-B iNat d'origine
(la LoRA du SSL est initialisée à zéro ⇒ teacher ≡ SimB-iNat). Il sert aussi de
**contrôle de validité** : il doit reproduire 0.4717 / 0.4931 / 0.5059 à ±0.001.

## 3. Pipelines

```
NARVAL (GPU, forward-only)                        LOCAL (CPU, mono-thread)
──────────────────────────                        ────────────────────────
leo_ssl/runs/…_seed{0,1,2}/checkpoints/ep*.pth    results/leo_explora_ssl/
checkpoints/simdinov2_vitb_inat21plantae.pth            │
tiles.zip + context_512.zip (+ _valtest)                │
        │                                               │
  1. sbatch scripts/slurm_leo_ssl_arctic_extract.sh     │
        │  (array 0-2 = seeds ; 1 extraction/ckpt)      │
        └──────── sig_embeddings/<tag>/ ──rsync────────►│
             {train,val,test}_{tile,ctx,labels}.npy     │
                                                        ▼
                                          2. python scripts/leo_ssl_alignment_probe.py
                                                        │
                                          3. python scripts/leo_ssl_alignment_plot.py
```

## 4. Commandes

### Étape 1 — extraction Narval

```bash
# Vérifier les pré-requis (une fois) :
ls $SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed0/checkpoints/
ls -la $SCRATCH/tiles.zip $SCRATCH/context_512.zip $SCRATCH/context_512_valtest.zip
ls -la $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth

cd ~/benchmark-memoire && git pull
mkdir -p logs
sbatch scripts/slurm_leo_ssl_arctic_extract.sh          # array 0-2 = 3 seeds
# test d'un seul seed :  SEED=0 sbatch scripts/slurm_leo_ssl_arctic_extract.sh
```

Chaque invocation traite **un** checkpoint → ~10-20 min (ViT-B, batch 128, fp32) ;
~10 checkpoints/seed ⇒ ~3 h/seed. Repartable : un tag complet est sauté, un job
ressoumis reprend où il en était. `last.pth` n'est utilisé que si aucun `ep*.pth`
n'existe (run coupé avant la 1re sauvegarde) — sinon il est ignoré (doublon du
dernier `ep*.pth`, cf. `ssl_train.py`).

### Étape 2 — rapatriement + sondes locales

```bash
rsync -avP narval:$SCRATCH/leo_ssl/arctic_probe/sig_embeddings/ \
    results/leo_explora_ssl/sig_embeddings/
python scripts/leo_ssl_alignment_probe.py --workers 4                 # sonde canonique par seed
python scripts/leo_ssl_alignment_probe.py --ensemble                  # + moyenne des seeds
# régénérer les tables sans recalculer :  --summary-only
```

### Étape 3 — figure

```bash
python scripts/leo_ssl_alignment_plot.py
```

## 5. Sorties

| Fichier | Contenu |
|---|---|
| `results/leo_explora_ssl/probes/<tag>_<variant>.json` | une sonde (tag × {tile,ctx,fused}) : F1 val/test, 8cls, bacc, best_C, dim, provenance |
| `results/leo_explora_ssl/probe_ensembles.json` | (`--ensemble`) moyenne des probas des seeds par époque — **diagnostic, non canonique (§4.4)** |
| `results/leo_explora_ssl/alignment_summary.csv` | une ligne par (checkpoint, variante) |
| `results/leo_explora_ssl/alignment_summary.md` | point époque 0, trajectoire moyenne ± σ sur 3 seeds, ensemble, rappel des références |
| `results/leo_explora_ssl/figures/alignment_doseresponse.{png,pdf}` | F1 vs époques de SSL, 3 courbes ± σ, références en pointillés |

## 6. Pièges respectés / connus

- **Aucune fuite** : les orthos Léo (`clairiere/maison/trail/rousseau`) ne sont pas
  dans les 38 orthos Arctic-TVC (vérifié) ; le SSL ne voit jamais les tuiles Arctic.
- **Split** : `spatial_datacurve/splits/frac100_seed0` — train **set-identique** aux 3
  seeds et val/test **md5-identiques** au split canonique `splits/` (vérifié le
  2026-10-04), donc les chiffres sont directement comparables au chapitre contexte.
- **Grille C identique** aux deux côtés de chaque comparaison (AGENTS.md §4.3) ;
  BLAS **mono-thread** pour toute sonde (§4.8) ; σ inter-seed ≈ 0.008 ⇒ aucune
  conclusion sous ~0.01 sans IC bootstrap (§4.4).
- **Ensemble de seeds** (`--ensemble`) : moyenne des probabilités des checkpoints de
  même époque. C'est l'**outil d'annotation** (décision par argmax) et un diagnostic de
  diversité des seeds, **pas** un chiffre de manuscrit — la fonction locale est copiée
  de la sonde canonique et un garde-fou vérifie que les deux donnent des métriques
  identiques.
- **`best_C` asymétrique** attendu fused (1e-4) vs tile/ctx (1e-3) — déjà un caveat
  documenté du chapitre contexte, il se reproduira ici.
- **`--amp`** change les features à ~1e-3 : laisser OFF pour reproduire au plus près
  les références ; l'utiliser seulement pour un balayage exploratoire rapide.
- **Prochaine étape (non implémentée)** : le bras *LoRA supervisé* (fine-tuner
  l'Arctic-TVC depuis un checkpoint SSL donné, puis re-sonder gelé) demandera une
  config dédiée pointant `checkpoint:` sur le `.pth` ExPLoRA — à faire seulement si
  la trajectoire gelée montre une tendance exploitable.

## 7. Fichiers livrés

| Fichier | Rôle |
|---|---|
| [`leo_ssl_extract_arctic.py`](leo_ssl_extract_arctic.py) | extraction frozen tuile + contexte par checkpoint (format SimDINOv2/ExPLoRA) |
| [`slurm_leo_ssl_arctic_extract.sh`](slurm_leo_ssl_arctic_extract.sh) | job Narval array 0-2 (seeds), repartable |
| [`leo_ssl_alignment_probe.py`](leo_ssl_alignment_probe.py) | sondes canoniques tile/ctx/fused + tables |
| [`leo_ssl_alignment_plot.py`](leo_ssl_alignment_plot.py) | figure dose-réponse |
