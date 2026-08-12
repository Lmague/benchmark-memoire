#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Ablation du rang LoRA — DINOv3 ViT-B/16 LVD, fraction=100%, 3 seeds.
#
# 5 rangs (2, 4, 8, 16, 32) × 3 seeds = 15 runs. Array 0-4, un rang par tâche,
# 3 seeds séquentiels sur la même slice A100 MIG 3g.20gb.
# Configs : configs/dinov3_vitb16_lvd_lora_r{R}.yaml  (convention alpha=2r, scaling=2.0)
#
# run_dir scopé par r ET alpha (modif datacurve_one_run.py : tag = {model}_{regime}_r{R}_a{A}_frac100_seed{N})
# → aucune collision possible entre rangs ni avec les runs existants (tags sans _r/_a).
# L'alpha est encodé dans le tag : si la convention change (alpha=2r → alpha=r par ex.),
# les tags changent → pas de re-écrasement, mais penser à supprimer les runs alpha=2r
# avant de relancer, ou utiliser un OUT_DIR distinct.
#
# Durée : ~1h20-1h30/seed (mesure indirecte : ViT-L 356.5 min/seed, slurm_lora_dinov3_vitl.sh:24)
#   → ~4h-4h30/tâche (3 seeds). --time=8:00:00 (marge ×1.8).
# Checkpoints → $SCRATCH/sota_screening/lora_rank_ablation/checkpoints/
# Embeddings  → $SCRATCH/sota_screening/lora_rank_ablation/embeddings/
# Runs        → $SCRATCH/sota_screening/lora_rank_ablation/runs/dinov3_vitb16_lvd_lora_r{R}a{A}_frac100_seed{0,1,2}/
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip (tuiles Arctic-TVC)
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#   - modif datacurve_one_run.py (tag _r{R}_a{A}) APPLIQUÉE — sinon collision (cf. rapport)
#
# Soumission : sbatch scripts/slurm_lora_rank_ablation.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_rank_abl
#SBATCH --array=0-4
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_rank_abl_%A_%a.out
#SBATCH --error=logs/lora_rank_abl_%A_%a.err
#SBATCH --account=def-bouguess_gpu

RANKS=(2 4 8 16 32)
R="${RANKS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="configs/dinov3_vitb16_lvd_lora_r${R}.yaml"
FRAC=1.00

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/lora_rank_ablation"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "Ablation rang | r=$R"
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

# Copie horodatée du config réellement utilisé — retrouver les hyperparamètres exacts
# sans ambiguïté plus tard (pratique de slurm_datacurve_lora.sh, évite le piège alpha).
cp "$CONFIG" "$OUT_DIR/config_used_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_r${R}.yaml"

# 3 seeds séquentielles sur la même A100
for SEED in 0 1 2; do
    echo ""
    echo "─── r=$R seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] r=$R seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID (r=$R) terminée."
echo "  Résultats   : $OUT_DIR/runs/dinov3_vitb16_lvd_lora_r${R}_frac100_seed{0,1,2}/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
