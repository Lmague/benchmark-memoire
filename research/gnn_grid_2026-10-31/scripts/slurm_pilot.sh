#!/bin/bash
# Pilote « agrégation relationnelle vs contexte fixe » — Arctic-TVC, 17 orthos.
# Préflight exigé par GOAL_SPEC §3/§9f : données, env, script, 1 tranche GPU, walltime,
# disque, partition. Une seule tranche a100_3g.20gb, jamais un nœud entier.
#
#SBATCH --job-name=gnn_grid_pilot
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
# partition laissée au scheduler (gpubase_interac refusait mem=48G sur tranche 3g)
#SBATCH --output=/scratch/lmague/gnn_grid_2026-10-31/logs/pilot_%j.out
#SBATCH --error=/scratch/lmague/gnn_grid_2026-10-31/logs/pilot_%j.err

set -euo pipefail

module load python/3.11
source "$HOME/ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0 TOKENIZERS_PARALLELISM=false

WORK=/scratch/lmague/gnn_grid_2026-10-31
CODE=$WORK/code
OUT=$WORK/pilot/out
FEAT=$WORK/pilot/features
CKPT=/scratch/lmague/checkpoints/simdinov2_vitb_inat21plantae.pth
mkdir -p "$OUT" "$FEAT" "$WORK/logs"

echo "=== PREFLIGHT $(date -u +%FT%TZ) ==="
hostname; echo "SLURM_TMPDIR=$SLURM_TMPDIR"
python -c "import torch,sklearn,numpy,transformers;print('torch',torch.__version__,'sk',sklearn.__version__,'np',numpy.__version__);print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
test -f "$CKPT" || { echo "FATAL checkpoint absent"; exit 21; }
test -f "$WORK/out/pilot_split.json" || { echo "FATAL split absent"; exit 22; }
df -h "$SLURM_TMPDIR" | tail -1
AVAIL_GB=$(df --output=avail -BG "$SLURM_TMPDIR" | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 25 ] || { echo "FATAL espace temporaire insuffisant: ${AVAIL_GB}G"; exit 23; }

# --- 1. staging des 17 orthos (tuiles brutes + crops 512) ---
ORTHOS="20230724_alder25_m3m 20230724_alder4_m3m 20230724_birch26_m3m \
20230724_birch30_m3m 20230724_birch54_m3m 20230724_lichentundra35_m3m \
20230724_lichentundra41_m3m 20230724_riparianshrub23_m3m 20230724_riparianshrub36_m3m \
20230724_sedge_mosstundra22_m3m 20230724_sedge_mosstundra34_m3m 20230725_birch2_m3m \
20230725_sedge_mosstundra19_m3m 20230725_tussockfield_m3m 20230728_birch42_m3m \
20230728_lichentundra44_m3m 20230728_tussockfield51_m3m"

P_TILES=(); P_CTX=()
for o in $ORTHOS; do
  P_TILES+=("tiles/arctic_vegetation/$o/*")
  P_CTX+=("context_512/arctic_vegetation/$o/*")
done

echo "=== STAGING $(date -u +%FT%TZ) ==="
unzip -q -o "$SCRATCH/tiles.zip" "${P_TILES[@]}" -d "$SLURM_TMPDIR"
unzip -q -o "$SCRATCH/context_512.zip" "${P_CTX[@]}" -d "$SLURM_TMPDIR"
NT=$(find "$SLURM_TMPDIR/tiles/arctic_vegetation" -name '*.png' | wc -l)
NC=$(find "$SLURM_TMPDIR/context_512/arctic_vegetation" -name '*.png' | wc -l)
echo "staged tiles=$NT ctx=$NC (attendu 30807 / 30807)"
[ "$NT" -eq 30807 ] || { echo "FATAL tuiles manquantes"; exit 31; }
[ "$NC" -eq 30807 ] || { echo "FATAL contextes manquants"; exit 32; }

# --- 2. extraction : test court puis complet ---
echo "=== EXTRACT smoke $(date -u +%FT%TZ) ==="
test -f "$WORK/coords_manifest.json" || { echo "FATAL coords_manifest absent"; exit 33; }

python "$CODE/extract_features.py" \
  --manifest "$WORK/coords_manifest.json" \
  --tiles-root "$SLURM_TMPDIR/tiles" --ctx-root "$SLURM_TMPDIR/context_512" \
  --out-dir "$WORK/pilot/smoke_extract" --repo "$HOME/benchmark-memoire" \
  --ckpt "$CKPT" --batch-size 64 --workers 4 --limit 48

python - <<'PY'
import json, numpy as np
p='/scratch/lmague/gnn_grid_2026-10-31/pilot/smoke_extract'
for v in ('tile','ctx512'):
    x=np.load(f'{p}/features_{v}.npy')
    assert x.shape==(48,768), (v,x.shape)
    assert np.isfinite(x).all(), v
    assert x.std(0).mean()>1e-3, (v,'features dégénérées')
print('SMOKE EXTRACT OK', )
PY

echo "=== EXTRACT full $(date -u +%FT%TZ) ==="
python "$CODE/extract_features.py" \
  --manifest "$WORK/coords_manifest.json" \
  --tiles-root "$SLURM_TMPDIR/tiles" --ctx-root "$SLURM_TMPDIR/context_512" \
  --out-dir "$FEAT" --repo "$HOME/benchmark-memoire" \
  --ckpt "$CKPT" --batch-size 256 --workers 8

# --- 3. bras + sonde + bootstrap ---
echo "=== ARMS $(date -u +%FT%TZ) ==="
python "$CODE/run_arms.py" \
  --split "$WORK/out/pilot_split.json" --features-dir "$FEAT" --out-dir "$OUT" \
  --repo "$HOME/benchmark-memoire" \
  --arms R0,R1,R2,R3,R4,L1,L2,L1p --seeds 0,1,2 \
  --boot 10000 --jobs 8 --perm-reps 5 --epochs 60 --patience 10 --tag pilot

# --- 4. persistance ---
rm -rf "$WORK/results_pilot"; mkdir -p "$WORK/results_pilot"
rsync -a "$OUT/" "$WORK/results_pilot/"
echo "=== DONE $(date -u +%FT%TZ) ==="
