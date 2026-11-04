#!/bin/bash
# GATv2 sur graphe complet (tuiles annotées + fenêtres auxiliaires), split canonique.
# CPU uniquement : les features sont pré-extraites, l'entraînement du GNN est léger.
#SBATCH --job-name=gnn_gatv2
#SBATCH --account=def-bouguess_cpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/gnn_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/gnn_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
# BLAS mono-thread : la sonde de MEAN9 doit rester canonique.
# torch a son propre pool, réglé par --threads dans le script.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

WORK=/scratch/lmague/gnn_grid_2026-10-31
OUT=$WORK/canon/gnn
mkdir -p "$OUT" "$WORK/logs"

echo "=== PREFLIGHT $(date -u +%FT%TZ) ==="
hostname
for f in "$WORK/out/canonical_split.json" "$WORK/canon/features/features_tile.npy" \
         "$WORK/canon/aux/aux_features.npy" "$WORK/canon/aux/aux_index.json"; do
  test -f "$f" || { echo "FATAL absent: $f"; exit 21; }
done
python - <<'PY'
import json, numpy as np
w='/scratch/lmague/gnn_grid_2026-10-31'
s=json.load(open(f'{w}/out/canonical_split.json'))['tiles']
x=np.load(f'{w}/canon/features/features_tile.npy', mmap_mode='r')
a=np.load(f'{w}/canon/aux/aux_features.npy', mmap_mode='r')
i=json.load(open(f'{w}/canon/aux/aux_index.json'))
assert x.shape[0]==len(s), (x.shape, len(s))
assert a.shape[0]==len(i), (a.shape, len(i))
print(f'OK tuiles={x.shape} aux={a.shape} dim_ok={x.shape[1]==a.shape[1]}')
PY

echo "=== GNN $(date -u +%FT%TZ) ==="
python "$WORK/code/run_gnn.py" \
  --split "$WORK/out/canonical_split.json" \
  --features "$WORK/canon/features/features_tile.npy" \
  --aux-features "$WORK/canon/aux/aux_features.npy" \
  --aux-index "$WORK/canon/aux/aux_index.json" \
  --out-dir "$OUT" \
  --arms MEAN9,GNN2,GNN3,GNN2rw \
  --seeds 0,1,2 --epochs 100 --hidden 128 --threads 8 \
  --repo "$HOME/benchmark-memoire"

rm -rf "$WORK/results_gnn"; mkdir -p "$WORK/results_gnn"
rsync -a "$OUT/" "$WORK/results_gnn/"
echo "=== DONE $(date -u +%FT%TZ) ==="
