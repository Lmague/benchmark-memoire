#!/bin/bash
# Reproduction du chiffre canonique tuile-seule (SimDINOv2-B) avec la sonde de l'étude.
# Hypothèse testée : l'écart pilote 0,3175 vs canonique 0,4717 vient du split, pas du code.
# Critère : si le job retrouve 0,4717 ± 0,002, la sonde est validée.
#
#SBATCH --job-name=gnn_repro_canon
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/repro_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/repro_%j.err

set -euo pipefail
module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

WORK=/scratch/lmague/gnn_grid_2026-10-31
cd "$WORK"
echo "=== REPRO $(date -u +%FT%TZ) sur $(hostname) ==="
python code/reproduce_canonical.py \
  --emb-dir "$WORK/emb" \
  --backbone simdinov2_vitb16 \
  --out "$WORK/repro_canonical.json"
echo "=== DONE $(date -u +%FT%TZ) ==="
