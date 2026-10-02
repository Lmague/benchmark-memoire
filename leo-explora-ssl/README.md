# Leo ExPLoRA-SSL

Extended pre-training auto-supervisé **vrai ExPLoRA** (Khanna et al.,
arXiv:2406.10973) : SimDINOv2 ViT-B/16 iNat-Plantae (photos de plantes au sol)
→ orthomosaïques aériennes Léo (nadir, 224px, 4 sites, ~355k tuiles).

Régime : backbone gelé sauf **LoRA Q,V (r=16) sur blocs 0-9 + 2 derniers blocs
full-FT + LayerNorms**, objectif **DINO + iBOT + KoLeo**, teacher EMA.
Multi-crop **2 vues globales distinctes + 6 locales** (vrai DINO — la v1 n'avait
qu'UNE globale partagée teacher/student). 3 seeds (array SLURM) → sorties
`..._seed{0,1,2}`, checkpoints `ep{NNN}.pth` + `last.pth`.
Checkpoint compatible `get_model.load_simdino_state_dict` (LoRA fusionnée).

## Pipeline

```
NARVAL (1x A100) — tout est déjà en place
───────────────────────────────
$SCRATCH/web/tiles + manifest_*.json     ← source canonique (355 486 tuiles),
$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth   = ceux des embeddings gelés
        │
        ├─ 3. sbatch run_narval.sbatch   (SSL ~10-20h × 3 seeds, array 0-2)
        │
        └─ 4. extract_probe_embeddings.py (gelé + adapté) ── .npz ──► LOCAL
                                                        5. eval_probe.py --probe
```

## 1. Tuiles (déjà en place — pas d'export)

Le SSL consomme **`$SCRATCH/web/tiles`** + `web/manifest_*.json`, le jeu canonique
des embeddings gelés (355 486 tuiles). Rien à exporter : `site_subdir: true` gère
le layout `web/tiles/{site}/{part}/{tr}/{idx}.jpg` (`f` du manifest = `{part}/{tr}/{idx}.jpg`).

> Optionnel : `export_ssl_tiles.py` produit des tuiles **brutes** (pixels capteur,
> layout plat `ssl_tiles/{site}/{idx}.jpg`, avec score `green` pour le
> sur-échantillonnage). À n'utiliser que si tu veux t'écarter des tuiles stretchées —
> alors `site_subdir: false`. ~15-20 Go à transférer.

## 2. Rien à transférer

`web/tiles`, les manifests et `$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth`
sont déjà sur Narval. Le préflight du sbatch vérifie leur présence.

## 3. Entraînement (Narval)

Édite `run_narval.sbatch` si besoin (`VENV=`, `BATCH_OVERRIDE=`), puis
`sbatch run_narval.sbatch`. Config : `configs/leo_vitb16_ssl.json`.
Array 0-2 = seeds 0,1,2. GPU : MIG `a100_3g.20gb` par défaut (file courte) ;
`BATCH_OVERRIDE=64`. Pour un A100 entier : éditer `--gres=gpu:a100:1` +
`BATCH_OVERRIDE=128`. Sorties :
`$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed{N}/checkpoints/`
(`ep{NNN}.pth` toutes les `save_every_epochs` + `last.pth`). On choisit l'époque
sur le probe des tuiles labellisées — pas `last` par défaut.

## 4-5. Éval probe (gelé vs adapté)

```bash
# local : liste des tuiles labellisées
python eval_probe.py --dump --leo /home/erazal/annotations_leo --out probe_labels.json
# NB : extract_probe_embeddings.py / eval_probe.py assument encore le layout plat
#      {site}/{idx}.jpg ; à adapter au layout web ({site}/{part}/{tr}/{idx}.jpg)
#      avant l'étape 4. NON bloquant pour l'entraînement (étape 3).
# Narval x2 (ckpt d'origine, puis un ep{NNN}.pth adapté — idéalement l'époque
# choisie sur le probe, pas forcément last) :
python extract_probe_embeddings.py --ckpt <ckpt> --tiles <dir_tuiles> \
    --labels probe_labels.json --out probe_emb_<frozen|adapted>.npz
# local : comparatif AUPRC holdout spatial
python eval_probe.py --probe --labels probe_labels.json \
    --frozen probe_emb_frozen.npz --adapted probe_emb_adapted.npz
```

Décision : on garde l'adapté s'il bat le gelé d'au moins +0.02 AUPRC par espèce.

## Contenu vendeur

`simdinov2/` : `models/`, `layers/`, `loss/` repris du fork DINOv2
Apache-2.0 (`_anciennes_experiences/vendors/sslplant`), inchangés —
garantit la compatibilité exacte des clés du checkpoint.
