#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA r=8 — SimDINOv2 ViT-B/16 × 3 seeds (pas d'array, job unique)
#
# Exécute 3 seeds séquentiellement sur une slice A100 3g.20gb.
# Checkpoints → $SCRATCH/sota_screening/lora_3models/checkpoints/ (fichiers préfixés ckpt_tag)
# Embeddings  → $SCRATCH/sota_screening/lora_3models/embeddings/
# Runs        → $SCRATCH/sota_screening/lora_3models/runs/simdinov2_vitb16_lora_frac100_seed{0,1,2}/
#
# Les 3 seeds sont déjà complétés (migrés vers le chemin ci-dessus après le fix de
# collision run_dir dans datacurve_one_run.py — cf. commit associé). Ce script n'a
# PAS besoin d'être relancé pour ce modèle ; --out-dir/--skip-if-done pointent vers
# le tag ckpt_tag={model}_{regime}_frac{XXX}_seed{N} désormais utilisé par run_dir,
# donc si relancé par erreur, les 3 seeds seront correctement skippés (done déjà présent).
#
# PRÉ-REQUIS :
#   - $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth
#
# Soumission (uniquement si besoin de relancer) : sbatch scripts/slurm_lora_vitb.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_vitb
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
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

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# 3 seeds séquentielles — run_dir scopé par ckpt_tag (inclut $MODEL), donc sûr même
# si un autre modèle écrit en parallèle vers le même $OUT_DIR.
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
echo "  Résultats   : $OUT_DIR/runs/${MODEL}_lora_frac100_seed{0,1,2}/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
