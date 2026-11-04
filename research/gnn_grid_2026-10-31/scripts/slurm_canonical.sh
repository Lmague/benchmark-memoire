#!/bin/bash
# Étude « agrégation relationnelle vs contexte fixe » sur le SPLIT CANONIQUE (35 orthos).
# Reprise à la demande du PI : mêmes 8 bras, mêmes critères, mêmes contrôles que le pilote,
# mais sur splits/*_11cls.csv (80 091 tuiles, 11 orthos de test) pour être comparable à 0,5059.
#
# Préflight : données, env, script, UNE GPU (jamais un nœud), walltime, disque, partition.
#SBATCH --job-name=gnn_grid_canon
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/canon_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/canon_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false

WORK=/scratch/lmague/gnn_grid_2026-10-31
CODE=$WORK/code
OUT=$WORK/canon/out
FEAT=$WORK/canon/features
CKPT=/scratch/lmague/checkpoints/simdinov2_vitb_inat21plantae.pth
MAN=$WORK/coords/coords_manifest_35.json
SPLIT=$WORK/out/canonical_split.json
mkdir -p "$OUT" "$FEAT" "$WORK/logs" "$WORK/canon"

echo "=== PREFLIGHT $(date -u +%FT%TZ) ==="
hostname; echo "SLURM_TMPDIR=$SLURM_TMPDIR"
python -c "import torch,sklearn,numpy;print('torch',torch.__version__,'sk',sklearn.__version__,'np',numpy.__version__);print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
for f in "$CKPT" "$MAN" "$SPLIT"; do test -f "$f" || { echo "FATAL absent: $f"; exit 21; }; done
df -h "$SLURM_TMPDIR" | tail -1
AVAIL_GB=$(df --output=avail -BG "$SLURM_TMPDIR" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 40 ] || { echo "FATAL espace temporaire insuffisant: ${AVAIL_GB}G"; exit 23; }
python - <<PY
import json
m=json.load(open("$MAN"))["entries"]; s=json.load(open("$SPLIT"))
assert len(m)==len(s["tiles"]), (len(m),len(s["tiles"]))
print("manifeste",len(m),"split",len(s["tiles"]),s["report"]["test"]["n"],"test")
PY

# --- 1. staging complet (35 orthos retenus par le split) ---
echo "=== STAGING $(date -u +%FT%TZ) ==="
unzip -q -o "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR"
unzip -q -o "$SCRATCH/context_512.zip" -d "$SLURM_TMPDIR"
NT=$(find "$SLURM_TMPDIR/tiles/arctic_vegetation" -name '*.png' | wc -l)
NC=$(find "$SLURM_TMPDIR/context_512/arctic_vegetation" -name '*.png' | wc -l)
echo "staged tiles=$NT ctx=$NC (split demande 80088)"
[ "$NT" -ge 80088 ] || { echo "FATAL tuiles manquantes"; exit 31; }
[ "$NC" -ge 80088 ] || { echo "FATAL contextes manquants"; exit 32; }

# --- 2. extraction (smoke puis complet) ---
echo "=== EXTRACT smoke $(date -u +%FT%TZ) ==="
python "$CODE/extract_features.py" \
  --manifest "$MAN" --tiles-root "$SLURM_TMPDIR/tiles" --ctx-root "$SLURM_TMPDIR/context_512" \
  --out-dir "$WORK/canon/smoke_extract" --repo "$HOME/benchmark-memoire" \
  --ckpt "$CKPT" --batch-size 64 --workers 4 --limit 64
python - <<'PY'
import numpy as np
p='/scratch/lmague/gnn_grid_2026-10-31/canon/smoke_extract'
for v in ('tile','ctx512'):
    x=np.load(f'{p}/features_{v}.npy'); assert x.shape==(64,768), (v,x.shape)
    assert np.isfinite(x).all() and x.std(0).mean()>1e-3, v
print('SMOKE EXTRACT OK')
PY

echo "=== EXTRACT full $(date -u +%FT%TZ) ==="
python "$CODE/extract_features.py" \
  --manifest "$MAN" --tiles-root "$SLURM_TMPDIR/tiles" --ctx-root "$SLURM_TMPDIR/context_512" \
  --out-dir "$FEAT" --repo "$HOME/benchmark-memoire" \
  --ckpt "$CKPT" --batch-size 256 --workers 8

# --- 3. bras + sonde + bootstrap ---
echo "=== ARMS $(date -u +%FT%TZ) ==="
python "$CODE/run_arms.py" \
  --split "$SPLIT" --features-dir "$FEAT" --out-dir "$OUT" \
  --repo "$HOME/benchmark-memoire" \
  --arms R0,R1,R2,R3,R4,L1,L2,L1p --seeds 0,1,2 \
  --boot 10000 --jobs 8 --perm-reps 5 --epochs 60 --patience 10 --tag canonical

echo "=== METRICS 8cls $(date -u +%FT%TZ) ==="
cp "$SPLIT" "$OUT/canonical_split.json"
python "$CODE/add_8cls_metrics.py" --out-dir "$OUT"

# --- 4. persistance ---
rm -rf "$WORK/results_canon"; mkdir -p "$WORK/results_canon"
rsync -a "$OUT/" "$WORK/results_canon/"
echo "=== DONE $(date -u +%FT%TZ) ==="
