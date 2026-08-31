#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Matrice d'attribution de R2 — sonde tuile seule (768) ET fusionnée (1536) sur
# 4 modèles, INFÉRENCE SEULE (aucun entraînement), split spatial frac100 seed0.
#
# Question : le gain R2 (f1_macro_pres_test ≈ 0.508, features fusionnées) vient-il
# de l'entraînement spécial (distillation contexte→tuile) ou juste du fait de
# donner le contexte (1536 dims) à la sonde ?
#
# Matrice (tous les modèles évalués sur le MÊME split spatial seed0) :
#   ┌───────────────────────┬──────────┬──────────┐
#   │ modèle                │ tile 768 │ fused 1536│
#   ├───────────────────────┼──────────┼──────────┤
#   │ DINOv3-B gelé         │   ✓      │    ✓      │  ← contrôle 1
#   │ LoRA r=2 (ablation)   │   ✓      │    ✓      │  ← contrôle 2 (sans distillation)
#   │ R1 dA_tL seed0        │   ✓      │    ✓      │  ← entraîné par distillation
#   │ R2 dB_tL seed0        │   ✓      │   (déjà 0.5098 dans metrics.json) │
#   └───────────────────────┴──────────┴──────────┘
#
# Lecture :
#   - gelé/fused ≈ 0.508  → le contexte dans la sonde fait tout ; la distillation
#                           n'apporte rien (0.508 = borne « info », pas un gain
#                           d'entraînement).
#   - lora_r2/fused ≈ 0.508  → un simple LoRA (sans distillation) exploite déjà le
#                           contexte ; le 0.508 de R2 n'est pas dû à la distillation.
#   - R1/fused > gelé/fused  → l'entraînement (distillation) aide à exploiter le
#                           contexte même sans y être exposé à l'inférence.
#   - R2/tile ≈ R1/tile   → l'entraînement R2 n'a pas amélioré le backbone seul.
#   - R2/tile > R1/tile   → l'entraînement R2 (avec la tête concat) a amélioré le
#                           backbone lui-même (surprise intéressante).
#
# ⚠️ CAVEAT contrôle 2 : le checkpoint LoRA r=2 vient de l'ablation rangs
# (results/lora_rank_ablation_CANONICAL.json, entraîné sur le split ALÉATOIRE) —
# c'est le seul plain-LoRA r2a4 disponible. Le backbone est ici un extracteur de
# features : l'écart de split d'entraînement est un confound mineur, à mentionner.
#
# Sorties : $SCRATCH/context_distill/controls/fused_probe_{tag}_seed0.json
#
# PRÉ-REQUIS (déjà en place sur Narval) :
#   - $SCRATCH/tiles.zip, $SCRATCH/context_1024.zip (val/test inclus)
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#   - $SCRATCH/context_distill/checkpoints/
#       dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100_seed0_best.pth   (R1)
#       dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed0_best.pth   (R2)
#   - $SCRATCH/sota_screening/lora_rank_ablation/checkpoints/
#       dinov3_vitb16_lvd_lora_r2a4_frac100_seed0_best.pth                       (contrôle 2)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_fused_matrix
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/context_fused_matrix_%j.out
#SBATCH --error=logs/context_fused_matrix_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE="${1:-1024}"
SEED="${2:-0}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
CTX_CKPT_DIR="$OUT_DIR/checkpoints"
LORA_R2_CKPT="$SCRATCH/sota_screening/lora_rank_ablation/checkpoints/dinov3_vitb16_lvd_lora_r2a4_frac100_seed${SEED}_best.pth"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_fused_matrix | context=$CONTEXT_SIZE seed=$SEED | gelé + LoRA-r2 + R1 + R2"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

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
if [[ -f "$CONTEXT_ZIP" ]]; then
    echo "[slurm] Extraction contexte ($CONTEXT_ZIP) → $SLURM_TMPDIR ..."
    unzip -q "$CONTEXT_ZIP" -d "$SLURM_TMPDIR/"
else
    echo "[ERROR] $CONTEXT_ZIP introuvable"; exit 1
fi
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ ! -d "$CONTEXT_DIR" ]] && { echo "[ERROR] $CONTEXT_DIR absent après unzip"; exit 1; }

SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
if [[ ! -f "$SPATIAL_CSV_DIR/test.csv" ]]; then
    echo "[ERROR] split spatial absent : $SPATIAL_CSV_DIR/test.csv"; exit 1
fi

CFG_OVERRIDE="$SLURM_TMPDIR/cfg_fused_matrix_seed${SEED}.yaml"
cat "$CONFIG" > "$CFG_OVERRIDE"
cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_fused_matrix.sh ─────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

run_probe() {  # $1=tag  $2=reps  $3..=args supplémentaires (--frozen | --ckpt-path X)
    local TAG="$1"; local REPS="$2"; shift 2
    echo ""
    echo "─── $TAG (reps=$REPS) ───"
    python scripts/context_fused_probe_controls.py \
        --config "$CFG_OVERRIDE" \
        --context-dir "$CONTEXT_DIR" \
        --out-dir "$OUT_DIR" \
        --seed "$SEED" \
        --tag "$TAG" \
        --reps "$REPS" \
        "$@"
    if [[ $? -ne 0 ]]; then
        echo "[WARN] $TAG échoué — continuation" >&2
    fi
}

# ── Contrôle 1 : DINOv3-B GELÉ (fusion pure, aucun entraînement) ──
run_probe frozen both --frozen

# ── Contrôle 2 : LoRA r=2 plain (ablation rangs, SANS distillation) ──
if [[ -f "$LORA_R2_CKPT" ]]; then
    run_probe lora_r2a4 both --ckpt-path "$LORA_R2_CKPT"
else
    echo "[WARN] checkpoint LoRA r2 absent : $LORA_R2_CKPT — contrôle 2 sauté" >&2
fi

# ── R1 : Design A teacher L seed0 — entraîné par distillation, sondé tuile + fusionné ──
R1_CKPT="$CTX_CKPT_DIR/dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100_seed${SEED}_best.pth"
if [[ -f "$R1_CKPT" ]]; then
    run_probe r1_dA_tL both --ckpt-path "$R1_CKPT"
else
    echo "[WARN] checkpoint R1 absent : $R1_CKPT" >&2
fi

# ── R2 : Design B teacher L seed0 — tuile SEULE (fused déjà connu = 0.5098) ──
R2_CKPT="$CTX_CKPT_DIR/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed${SEED}_best.pth"
if [[ -f "$R2_CKPT" ]]; then
    run_probe r2_dB_tL tile --ckpt-path "$R2_CKPT"
else
    echo "[WARN] checkpoint R2 absent : $R2_CKPT" >&2
fi

echo ""
echo "[slurm] matrice terminée → $OUT_DIR/controls/"
ls -la "$OUT_DIR/controls/" 2>/dev/null
