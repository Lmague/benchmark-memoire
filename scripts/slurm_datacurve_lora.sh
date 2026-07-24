#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Data curve LoRA r=8 — DINOv3-B LoRA pur (Q,V sur tous les blocs)
#
# 8 fractions × 3 seeds = 24 runs. Array de 8 jobs, 3 seeds séquentiels chacun.
# Fractions : 0.5%, 1%, 5%, 10%, 25%, 50%, 70%, 100%
#
# Soumission : sbatch scripts/slurm_datacurve_lora.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_dc
#SBATCH --array=0-7
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/lora_dc_%A_%a.out
#SBATCH --error=logs/lora_dc_%A_%a.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/dinov3_lora_datacurve.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/lora"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

FRACS=(0.005 0.01 0.05 0.10 0.25 0.50 0.70 1.00)
FRAC="${FRACS[$SLURM_ARRAY_TASK_ID]}"
PCT=$(python3 -c "print(int(round(${FRAC}*100)))")

echo "═══════════════════════════════════════════════"
echo "LoRA ViT-B DINOv3 | array=$SLURM_ARRAY_TASK_ID → frac=$FRAC (${PCT}%)"
echo "Nœud : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
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

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# 3 seeds séquentiels sur la même A100
for SEED in 0 1 2; do
    echo ""
    echo "─── LoRA frac=$FRAC seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] frac=$FRAC seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID (frac=$FRAC) terminée."
echo "  Résultats : $OUT_DIR/runs/"
echo "  Embeddings : $OUT_DIR/embeddings/"
