#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA r=8 — 2 × 3 seeds = 6 runs (array 0-1)
#
# Task 0 : DINOv3 ViT-L/16 LVD
# Task 1 : SimDINOv2 ViT-L/16
#
# Chaque task exécute 3 seeds séquentiellement sur la même slice A100 2g.20gb.
# Checkpoints → $SCRATCH/sota_screening/lora_3models/checkpoints/{model}/
# Embeddings → $SCRATCH/sota_screening/lora_3models/embeddings/
# Runs       → $SCRATCH/sota_screening/lora_3models/runs/
#
# PRÉ-REQUIS :
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitl16-pretrain-lvd1689m/ (DINOv3-L)
#   - $SCRATCH/checkpoints/simdinov2_vitl_inat21plantae.pth  (SimDINOv2-L)
#
# ⚠ MIG slices (partition mig) : à vérifier avec `sinfo | grep mig`.
#   Si indisponible, remplacer `--gres=gpu:a100:2g.20gb` par `--gres=gpu:a100:1`
#   et `--mem` par 60G.
#
# Soumission : sbatch scripts/slurm_lora_vitl.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_vitl
#SBATCH --array=0-1
#SBATCH --gres=gpu:a100:2g.20gb
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora_vitl_%A_%a.out
#SBATCH --error=logs/lora_vitl_%A_%a.err
#SBATCH --account=def-bouguess_gpu

CONFIGS=(
    "configs/dinov3_vitl16_lvd_lora.yaml"
    "configs/simdinov2_vitl16_lora.yaml"
)

MODEL_NAMES=(
    "dinov3_vitl16_lvd"
    "simdinov2_vitl16"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"
FRAC=1.00

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/lora_3models"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "LoRA r=8 | array=$SLURM_ARRAY_TASK_ID → $MODEL"
echo "Config  : $CONFIG"
echo "Nœud    : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# Tuiles → $SLURM_TMPDIR
echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles extraites."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

mkdir -p "$OUT_DIR/runs/$MODEL" "$OUT_DIR/checkpoints/$MODEL" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# 3 seeds séquentielles
for SEED in 0 1 2; do
    echo ""
    echo "─── $MODEL seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] $MODEL seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID ($MODEL) terminée."
echo "  Résultats   : $OUT_DIR/runs/$MODEL/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
