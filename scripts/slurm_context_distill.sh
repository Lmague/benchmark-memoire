#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Self-distillation contexte→tuile — DINOv3-B LoRA student + DINOv3-L teacher gelé.
#
# 1 GPU A100 40 Go COMPLET (--gres=gpu:a100:1, PAS de nœud entier, PAS de slice MIG —
# le teacher ViT-L (forward seul, gelé, pas d'état optimizer) + le student ViT-B LoRA
# (peu de params entraînables) tiennent sur un A100 complet mais dépassent les slices
# MIG 10-20 Go utilisées par les autres jobs LoRA de ce dépôt).
#
# ⚠️ --time NON MESURÉ (aucun run GPU exécuté — contrainte de la mission : "ne lancer
# aucun entraînement"). Estimation prudente extrapolée de slurm_lora_dinov3_vitl.sh
# (LoRA r8 ViT-L seul = 356.5 min/seed mesuré, job 66522732) + surcoût forward teacher
# ViT-L à chaque step + I/O tuile+contexte doublé → --time=24:00:00. À AJUSTER après le
# premier run réel (regarder logs/context_distill_*.out, ne pas resoumettre à l'aveugle
# si le job time-out avant la fin).
#
# ⚠️ --mem UNIFORME (64G) pour R1/R2/R3, PAS de bump à 128G pour 2048px : contrairement
# à une fenêtre de contexte native, context_crop.py REDIMENSIONNE tout contexte à
# out-size=224 AVANT sauvegarde (cf. docstring de context_crop.py) — le tenseur chargé
# en entraînement a donc la MÊME taille (224×224) quel que soit --context-size. Pas de
# surcoût mémoire différentiel entre R1 (1024px) et R3 (2048px) sous ce design.
#
# Usage : CE SCRIPT PREND $1 = seed (0, 1 ou 2) ET $2 = context_size EN OPTION
# (défaut 1024 = R1). Soumettre par seed (permet aux 3 seeds de tourner EN PARALLÈLE
# sur Narval plutôt que séquentiellement — contrairement aux autres scripts du dépôt,
# chaque run ici implique un forward ViT-L teacher supplémentaire à chaque step, donc
# plus coûteux ; la parallélisation inter-seeds est préférable ici).
#
# Ordre recommandé (cf. scripts/context_distill_README.md) :
#   1. R1 (1024px, les 3 seeds) :
#        sbatch scripts/slurm_context_distill.sh 0
#        sbatch scripts/slurm_context_distill.sh 1
#        sbatch scripts/slurm_context_distill.sh 2
#   2. SI R1 bat la baseline (dinov3_vitb16_lvd_lora_r8_published F1=0.4835±0.0011,
#      results/lora_rank_ablation_CANONICAL.json — sur split ALÉATOIRE, à comparer ici
#      au frac100 spatial : F1=0.4827±0.0042 (3 seeds, mesuré 2026-08-29 depuis
#      results/spatial_datacurve_CANONICAL.csv, cf. scripts/context_distill_README.md §3) → R2/R3 :
#        sbatch scripts/slurm_context_distill.sh 0 512
#        sbatch scripts/slurm_context_distill.sh 1 512
#        sbatch scripts/slurm_context_distill.sh 2 512
#        sbatch scripts/slurm_context_distill.sh 0 2048   # (+ seeds 1, 2)
#        sbatch scripts/slurm_context_distill.sh 1 2048
#        sbatch scripts/slurm_context_distill.sh 2 2048
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip                      (tuiles 224px, convention existante)
#   - $SCRATCH/context_512.zip / _1024.zip / _2048.zip   (produits par
#     scripts/context_crop.py EN LOCAL puis transférés — cf. slurm_context_crop.sh)
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitl16-pretrain-lvd1689m/
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=context_distill
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/context_distill_%j.out
#SBATCH --error=logs/context_distill_%j.err
#SBATCH --account=def-bouguess_gpu

SEED="${1:?usage: sbatch scripts/slurm_context_distill.sh <seed:0|1|2> [context_size:512|1024|2048]}"
CONTEXT_SIZE="${2:-1024}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
SPATIAL_CSV_DIR="$CODE_DIR/splits_spatial/frac100_seed${SEED}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "context_distill | seed=$SEED context_size=$CONTEXT_SIZE"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -f "$SPATIAL_CSV_DIR/train.csv" ]] && {
    echo "[ERROR] split spatial absent: $SPATIAL_CSV_DIR/train.csv (scripts/make_spatial_splits.py)"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs

echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

CONTEXT_ZIP="$SCRATCH/context_${CONTEXT_SIZE}.zip"
[[ ! -f "$CONTEXT_ZIP" ]] && {
    echo "[ERROR] $CONTEXT_ZIP introuvable — lancer context_crop.py EN LOCAL et transférer"
    echo "        les context_<size>.zip d'abord (cf. scripts/slurm_context_crop.sh)."
    exit 1
}
echo "[slurm] Extraction contexte ($CONTEXT_ZIP) → $SLURM_TMPDIR ..."
unzip -q "$CONTEXT_ZIP" -d "$SLURM_TMPDIR/"
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ ! -d "$CONTEXT_DIR" ]] && { echo "[ERROR] $CONTEXT_DIR absent après unzip"; exit 1; }

# Override généré : pointe csv_dir vers le split spatial de ce seed ET tiles_dir vers
# le SLURM_TMPDIR (même mécanique que slurm_datacurve_spatial.sh:90-98).
CFG_OVERRIDE="$SLURM_TMPDIR/cfg_context_distill_seed${SEED}_ctx${CONTEXT_SIZE}.yaml"
cat "$CONFIG" > "$CFG_OVERRIDE"
cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_distill.sh ───────────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

mkdir -p "$OUT_DIR"
cp "$CFG_OVERRIDE" "$OUT_DIR/config_used_${SLURM_JOB_ID}_seed${SEED}_ctx${CONTEXT_SIZE}.yaml"

python scripts/context_distill.py \
    --config "$CFG_OVERRIDE" \
    --teacher dinov3_vitl16_lvd \
    --context-size "$CONTEXT_SIZE" \
    --context-dir "$CONTEXT_DIR" \
    --design A \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --skip-if-done

echo ""
echo "[slurm] seed=$SEED context_size=$CONTEXT_SIZE terminé."
echo "  Résultats : $OUT_DIR/runs/dinov3_vitb16_lvd_ctxdistill_ctx${CONTEXT_SIZE}_r2a4_frac100_seed${SEED}/metrics.json"
