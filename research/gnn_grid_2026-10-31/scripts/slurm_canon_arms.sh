#!/bin/bash
# Étape B — bras, sondes et bootstrap sur le split canonique (CPU seulement).
# Dépend de l'étape A (features extraites sur $SCRATCH).
#SBATCH --job-name=gnn_canon_arms
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/ca_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/ca_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

WORK=/scratch/lmague/gnn_grid_2026-10-31
OUT=$WORK/canon/out
FEAT=$WORK/canon/features
SPLIT=$WORK/out/canonical_split.json
mkdir -p "$OUT" "$WORK/logs"

echo "=== PREFLIGHT $(date -u +%FT%TZ) ==="
hostname
for f in "$FEAT/features_tile.npy" "$FEAT/features_ctx512.npy" "$FEAT/index.json" "$SPLIT"; do
  test -f "$f" || { echo "FATAL absent: $f"; exit 21; }
done
python - <<'PY'
import json, numpy as np
f='/scratch/lmague/gnn_grid_2026-10-31/canon/features'
i=json.load(open(f'{f}/index.json'))
n=len(i['keys']) if isinstance(i,dict) and 'keys' in i else len(i.get('order',[]))
for v in ('tile','ctx512'):
    x=np.load(f'{f}/features_{v}.npy'); print(v,x.shape,x.dtype)
    assert x.shape[1]==768 and np.isfinite(x).all()
    assert x.shape[0]==n, (x.shape[0],n)
print('FEATURES OK',n)
PY

echo "=== ARMS $(date -u +%FT%TZ) ==="
python "$WORK/code/run_arms.py" \
  --split "$SPLIT" --features-dir "$FEAT" --out-dir "$OUT" \
  --repo "$HOME/benchmark-memoire" \
  --arms R0,R1,R2,R3,R4,L1,L2,L1p --seeds 0,1,2 \
  --boot 10000 --jobs 8 --perm-reps 5 --epochs 60 --patience 10 --tag canonical

echo "=== METRICS 8cls $(date -u +%FT%TZ) ==="
cp "$SPLIT" "$OUT/canonical_split.json"
python "$WORK/code/add_8cls_metrics.py" --out-dir "$OUT"

rm -rf "$WORK/results_canon"; mkdir -p "$WORK/results_canon"
rsync -a "$OUT/" "$WORK/results_canon/"
echo "=== DONE $(date -u +%FT%TZ) ==="
