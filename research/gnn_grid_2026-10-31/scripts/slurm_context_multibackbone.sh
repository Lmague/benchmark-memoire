#!/bin/bash
#SBATCH --account=def-bouguess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:30:00
#SBATCH --job-name=ctx_mb
#SBATCH --output=%x_%j.out

# Context + mean-field smoothing on several backbones, same canonical split.
# Each backbone: build features in canonical order, run the R0/R3 probe with
# mean-field and ICM, then paired bootstrap over maps for the three contrasts.
#
# Preflight: 8 backbones, 8 CPUs, one process per backbone, BLAS mono-thread
# (decoding_ablation forces it). Memory: worst case ~2 GB per process.
set -euo pipefail

module load python/3.11
PY=${PY:-$HOME/ENV/bin/python}
WORK=${WORK:-$SCRATCH/context_multibackbone}
CODE=$WORK/code
SPLIT=$WORK/canonical_split.json
REPO=$WORK/repo
OUT=$WORK/out
mkdir -p "$OUT"

# Extended grids: the previous study selected beta=12 (max) and iters=5 (max).
MFB="0.1,0.25,0.5,1,2,3,5,8,12,16"
MFI="1,2,3,5,8,12"

run_one() {
  set -e
  name=${1%%|*}; emb=${1#*|}
  o=$OUT/$name; mkdir -p "$o"
  echo "[$name] features from $emb"
  $PY "$CODE/build_backbone_features.py" --name "$name" --emb-dir "$emb" --split "$SPLIT" --out-dir "$o"
  echo "[$name] decoding + smoothing"
  $PY "$CODE/decoding_ablation.py" --split "$SPLIT" --features "$o/features_tile.npy" \
      --out "$o/ablation.json" --repo "$REPO" \
      --mf-beta "$MFB" --mf-iters "$MFI" --icm-beta "$MFB" --icm-iters "$MFI"
  for pair in "R0_base R0_MF" "R3_base R3_MF" "R0_base R3_base"; do
    set -- $pair
    $PY "$CODE/compare_arms.py" --split "$SPLIT" --preds-dir "$o/preds" \
        --arm-a "$1" --arm-b "$2" --boot 10000 --out "$o/cmp_$1_$2.json"
  done
  echo "[$name] done"
}

JOBS=(
  "simdinov2_vitb16|$SCRATCH/embeddings"
  "dinov3_vitb16_lvd|$SCRATCH/embeddings"
  "dinov3_vitl16_lvd|$SCRATCH/embeddings"
  "dinov3_vith16plus_lvd|$SCRATCH/embeddings"
  "simdinov2_vitl16|$SCRATCH/embeddings"
  "dinov3_vitl16_lvd_lora_s0|$SCRATCH/sota_screening/lora_3models/embeddings/dinov3_vitl16_lvd_lora_frac100_seed0"
  "simdinov2_vitb16_lora_r8a16_s0|$SCRATCH/sota_screening/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a16_frac100_seed0"
)

export -f run_one
export PY CODE SPLIT REPO OUT MFB MFI

printf '%s\n' "${JOBS[@]}" | xargs -P 8 -I{} bash -c 'run_one "$@"' _ {}
echo "[ctx_mb] all done"
