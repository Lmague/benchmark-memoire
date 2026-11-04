#!/bin/bash
# Diagnostic A2 : L1/L2 entraînés 15 époques fixes, sans sélection sur validation.
# CPU seulement — réutilise les features en cache sur $SCRATCH (aucune ré-extraction).
#
#SBATCH --job-name=gnn_grid_diag15
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/diag15_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/diag15_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

WORK=/scratch/lmague/gnn_grid_2026-10-31
cd "$WORK"
echo "=== DIAG15 $(date -u +%FT%TZ) sur $(hostname) ==="
python code/run_arms.py \
  --split "$WORK/out/pilot_split.json" \
  --features-dir "$WORK/pilot/features" \
  --out-dir "$WORK/pilot/out_diag15" \
  --repo "$HOME/benchmark-memoire" \
  --arms L1,L2 --seeds 0,1,2 --boot 200 --jobs 8 --epochs 15 --fixed-epochs 15 \
  --tag diag15
rsync -a "$WORK/pilot/out_diag15/" "$WORK/results_diag15/"
echo "=== DONE $(date -u +%FT%TZ) ==="
