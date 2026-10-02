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
LOCAL (cette machine, CPU)            NARVAL (1x A100)
─────────────────────────             ─────────────────
1. export_ssl_tiles.py                3. sbatch run_narval.sbatch
   tuiles BRUTES 224px ─────zip───►      (SSL ~10-20h × 3 seeds, array)
   + manifests (score green)          4. extract_probe_embeddings.py x2
                                        (frozen + adapté)
2. eval_probe.py --dump               ──── .npz ───► LOCAL
   probe_labels.json ─────►           5. eval_probe.py --probe
```

## 1. Export (local)

```bash
~/miniconda3/bin/python3 export_ssl_tiles.py              # ~355k tuiles, ~15-20 Go
~/miniconda3/bin/python3 export_ssl_tiles.py --site maison --limit 500   # test
```

⚠️ Pixels **bruts** (pas de stretch) : le SSL doit voir la distribution capteur.
Le score `green` (Excess Green) sert au sur-échantillonnage x4 des tuiles
végétalisées à l'entraînement (`green_upsample`).

## 2. Transfert Narval

```bash
tar -czf ssl_tiles.tar.gz -C data ssl_tiles/
scp ssl_tiles.tar.gz simdinov2_vitb_inat21plantae.pth narval:$SCRATCH/leo_ssl/
# sur Narval :
mkdir -p $SCRATCH/leo_ssl && cd $SCRATCH/leo_ssl && tar -xzf ssl_tiles.tar.gz
cp /chemin/manifests/ssl_manifest_*.json .
```

## 3. Entraînement (Narval)

Édite `run_narval.sbatch` si besoin (`VENV=`, `BATCH_OVERRIDE=`), puis
`sbatch run_narval.sbatch`. Config : `configs/leo_vitb16_ssl.json`.
Array 0-2 = seeds 0,1,2. Sorties :
`$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed{N}/checkpoints/`
(`ep{NNN}.pth` toutes les `save_every_epochs` + `last.pth`). On choisit l'époque
sur le probe des tuiles labellisées — pas `last` par défaut.

## 4-5. Éval probe (gelé vs adapté)

```bash
# local : liste des tuiles labellisées
python eval_probe.py --dump --leo /home/erazal/annotations_leo --out probe_labels.json
# Narval x2 (ckpt d'origine, puis un ep{NNN}.pth adapté — idéalement l'époque
# choisie sur le probe, pas forcément last) :
python extract_probe_embeddings.py --ckpt <ckpt> --tiles $SCRATCH/leo_ssl/ssl_tiles \
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
