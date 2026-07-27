#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA r=8 — SimDINOv2 ViT-B/16 × 3 seeds (pas d'array, job unique)
#
# Exécute 3 seeds séquentiellement sur une slice A100 1g.10gb.
# Checkpoints → $SCRATCH/sota_screening/lora_3models/checkpoints/simdinov2_vitb16/
# Embeddings → $SCRATCH/sota_screening/lora_3models/embeddings/
# Runs       → $SCRATCH/sota_screening/lora_3models/runs/
#
# PRÉ-REQUIS :
#   - $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth
#
# ⚠ MIG slices (partition mig) : à vérifier avec `sinfo | grep mig`.
#   Si indisponible, remplacer `--gres=gpu:a100:1g.10gb` par `--gres=gpu:a100:1`
#   et `--mem` par 60G.
#
# Soumission : sbatch scripts/slurm_lora_vitb.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_vitb
#SBATCH --gres=gpu:a100:1g.10gb
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/lora_vitb_%j.out
#SBATCH --error=logs/lora_vitb_%j.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/simdinov2_vitb16_lora.yaml"
MODEL="simdinov2_vitb16"
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
echo "[slurm] $MODEL terminé."
echo "  Résultats   : $OUT_DIR/runs/$MODEL/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
