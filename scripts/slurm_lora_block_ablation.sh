#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Ablation de POSITION LoRA — DINOv3 ViT-B/16 LVD, fraction=100%, 3 seeds.
#
# 2 variantes × 3 seeds = 6 runs. Array 0-1, une variante par tâche, 3 seeds
# séquentiels sur la même slice A100 MIG 3g.20gb.
#   r=0 : dernières 6 couches  (lora_block_indices: [6..11]) — hypothèse head-heavy,
#         motivée par la norme ||ΔW||/||W0|| par couche des checkpoints r8a16 de
#         l'ablation de rang (croissante en profondeur, pic bloc 11 ; Spearman
#         inter-seed 0.94-0.97).
#   r=1 : premières 6 couches ([0..5]) — contrôle : si ≈ « toutes » aussi, la
#         position ne compte pas ; si dégradé, l'adaptation est head-heavy.
#
# Configs : configs/dinov3_vitb16_lvd_lora_r8_b611.yaml / _b05.yaml
#   (r=8, α=16, mêmes hyperparams que la production ; tag run = {model}_lora_r8a16_b{...}_frac100_seed{N}).
# Durée : ~1h20-1h30/seed (cf. slurm_lora_rank_ablation.sh) → ~4h30/tâche.
#
# PRÉ-REQUIS : modif datacurve_one_run.py (tag _b{blocs}) APPLIQUÉE + src/config.py
#   (lora_block_indices) + src/models.py (injection restreinte) — validées le 2026-08-12
#   (build_model : b611 → 147,456 params LoRA, blocs {6..11}).
#
# Soumission : sbatch scripts/slurm_lora_block_ablation.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_blk_abl
#SBATCH --array=0-1
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_blk_abl_%A_%a.out
#SBATCH --error=logs/lora_blk_abl_%A_%a.err
#SBATCH --account=def-bouguess_gpu

VARIANTES=("b611" "b05")
TAG="${VARIANTES[$SLURM_ARRAY_TASK_ID]}"
CONFIG="configs/dinov3_vitb16_lvd_lora_r8_${TAG}.yaml"
FRAC=1.00

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/lora_block_ablation"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "Ablation position LoRA | variante=$TAG"
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

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

cp "$CONFIG" "$OUT_DIR/config_used_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}.yaml"

# 3 seeds séquentielles sur la même A100
for SEED in 0 1 2; do
    echo ""
    echo "─── $TAG seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] $TAG seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID ($TAG) terminée."
echo "  Résultats : $OUT_DIR/runs/dinov3_vitb16_lvd_lora_r8a16_b${TAG}_frac100_seed{0,1,2}/"
