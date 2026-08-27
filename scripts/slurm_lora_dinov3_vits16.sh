#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA r=8 — DINOv3 ViT-S/16 LVD × 3 seeds (job unique, pas d'array)
#
# Calqué sur scripts/slurm_lora_dinov3_vitl.sh (même famille DINOv3 HF, mêmes
# conventions : OUT_DIR, compte, demi-slice A100, HF offline). Seuls changent le
# modèle (ViT-S, dim 384, ~4× moins cher que ViT-B), les durées et le pré-requis HF.
#
# Checkpoints → $SCRATCH/sota_screening/lora_3models/checkpoints/
# Embeddings  → $SCRATCH/sota_screening/lora_3models/embeddings/
# Runs        → $SCRATCH/sota_screening/lora_3models/runs/dinov3_vits16_lvd_lora_r8a16_frac100_seed{0,1,2}/
#               (tag = {model}_lora_r8a16_frac100_seed{N} — r ET alpha encodés)
#
# PRÉ-REQUIS :
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vits16-pretrain-lvd1689m/
#
# Durée estimée : ViT-L seed=0 ≈ 356 min (mesuré, job 66522732) ; ViT-B ≈ 4× plus
# rapide ; ViT-S ≈ 4× plus rapide que ViT-B → ~20-30 min/seed → 3 seeds ≈ 1-2 h.
# --time=6:00:00 = large marge (comme slurm_lora_vitb.sh).
#
# Soumission : sbatch scripts/slurm_lora_dinov3_vits16.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_dinov3_vits16
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=logs/lora_dinov3_vits16_%j.out
#SBATCH --error=logs/lora_dinov3_vits16_%j.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/dinov3_vits16_lvd_lora8.yaml"
MODEL="dinov3_vits16_lvd"
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
echo "LoRA r=8 | $MODEL"
echo "Config  : $CONFIG"
echo "Nœud    : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID"
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

# 3 seeds séquentielles — run_dir scopé par ckpt_tag (inclut $MODEL + r8a16), donc sûr
# même si un autre modèle écrit en parallèle vers le même $OUT_DIR.
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
echo "[slurm] $MODEL terminé."
echo "  Résultats   : $OUT_DIR/runs/${MODEL}_lora_r8a16_frac100_seed{0,1,2}/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
