#!/bin/bash
# Étape A — staging + extraction des features (GPU), split canonique 32 orthos.
# Découpé pour backfiller plus facilement qu'un job de 2 h.
#SBATCH --job-name=gnn_canon_extract
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/cx_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/cx_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

WORK=/scratch/lmague/gnn_grid_2026-10-31
FEAT=$WORK/canon/features
MAN=$WORK/coords/coords_manifest_35.json
CKPT=/scratch/lmague/checkpoints/simdinov2_vitb_inat21plantae.pth
mkdir -p "$FEAT" "$WORK/logs"

echo "=== PREFLIGHT $(date -u +%FT%TZ) ==="
hostname; echo "SLURM_TMPDIR=$SLURM_TMPDIR"
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
for f in "$CKPT" "$MAN"; do test -f "$f" || { echo "FATAL absent: $f"; exit 21; }; done
AVAIL_GB=$(df --output=avail -BG "$SLURM_TMPDIR" | tail -1 | tr -dc '0-9')
echo "espace temporaire: ${AVAIL_GB}G"
[ "$AVAIL_GB" -ge 40 ] || { echo "FATAL espace insuffisant"; exit 23; }

echo "=== STAGING $(date -u +%FT%TZ) ==="
unzip -q -o "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR"
unzip -q -o "$SCRATCH/context_512.zip" -d "$SLURM_TMPDIR"
NT=$(find "$SLURM_TMPDIR/tiles/arctic_vegetation" -name '*.png' | wc -l)
NC=$(find "$SLURM_TMPDIR/context_512/arctic_vegetation" -name '*.png' | wc -l)
echo "staged tiles=$NT ctx=$NC"
[ "$NT" -ge 80088 ] || { echo "FATAL tuiles manquantes"; exit 31; }
[ "$NC" -ge 80088 ] || { echo "FATAL contextes manquants"; exit 32; }

echo "=== EXTRACT smoke $(date -u +%FT%TZ) ==="
python "$WORK/code/extract_features.py" \
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
python "$WORK/code/extract_features.py" \
  --manifest "$MAN" --tiles-root "$SLURM_TMPDIR/tiles" --ctx-root "$SLURM_TMPDIR/context_512" \
  --out-dir "$FEAT" --repo "$HOME/benchmark-memoire" \
  --ckpt "$CKPT" --batch-size 256 --workers 4
echo "=== DONE $(date -u +%FT%TZ) ==="
