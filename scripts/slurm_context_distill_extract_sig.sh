#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Extraction embeddings train/val/test — 9 runs context_distill (R1/R2/R3 × 3 seeds)
# pour le bootstrap canonique (significance_tier.py + make_tables.py).
#
# Sorties : $SCRATCH/context_distill/sig_embeddings/<tag>/{train,val,test}.npy
#           + *_labels.npy (11 classes, RHOL retirée) — à rapatrier en local dans
#           results/context_distill/sig_embeddings/ puis pointer via registry.
#
# Durée estimée : ~40 min/run (extraction seul, sans probe) × 9 ≈ 6-7 h → time 10 h.
# Léger en VRAM (forward seul, MIG a100_3g.20gb).
#
# PRÉ-REQUIS : checkpoints *_best.pth des 9 runs, tiles.zip, context_1024.zip
# (val/test inclus), hf_cache DINOv3-B.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_extract_sig
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/context_distill_extract_sig_%j.out
#SBATCH --error=logs/context_distill_extract_sig_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE="${1:-1024}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
CKPT_DIR="$OUT_DIR/checkpoints"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_extract_sig | context=$CONTEXT_SIZE | 9 runs (R1/R2/R3 × 3 seeds)"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs

echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
[[ -f "$SCRATCH/tiles.zip" ]] && unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/" \
    || { echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1; }
CONTEXT_ZIP="$SCRATCH/context_${CONTEXT_SIZE}.zip"
[[ -f "$CONTEXT_ZIP" ]] && unzip -q "$CONTEXT_ZIP" -d "$SLURM_TMPDIR/" \
    || { echo "[ERROR] $CONTEXT_ZIP introuvable"; exit 1; }
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ -d "$CONTEXT_DIR" ]] || { echo "[ERROR] $CONTEXT_DIR absent"; exit 1; }

# 3 configs × 3 seeds : (design, teacher, fused)
for SPEC in "A dinov3_vitl16_lvd" "A ema_self" "B dinov3_vitl16_lvd"; do
    read -r DESIGN TEACHER <<< "$SPEC"
    FUSED=""
    [[ "$DESIGN" == "B" ]] && FUSED="--fused"
    for SEED in 0 1 2; do
        case "$TEACHER" in
            dinov3_vitl16_lvd) TAG="dinov3_vitb16_lvd_ctxdistill_d${DESIGN}_tL_ctx${CONTEXT_SIZE}_r2a4_frac100_seed${SEED}" ;;
            ema_self)          TAG="dinov3_vitb16_lvd_ctxdistill_d${DESIGN}_tEMA_ctx${CONTEXT_SIZE}_r2a4_frac100_seed${SEED}" ;;
        esac
        CKPT="$CKPT_DIR/${TAG}_best.pth"
        [[ -f "$CKPT" ]] || { echo "[WARN] checkpoint absent : $CKPT — saut" >&2; continue; }

        SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
        [[ -f "$SPATIAL_CSV_DIR/test.csv" ]] || { echo "[WARN] split seed=$SEED absent" >&2; continue; }

        CFG_OVERRIDE="$SLURM_TMPDIR/cfg_extract_sig_seed${SEED}.yaml"
        cat "$CONFIG" > "$CFG_OVERRIDE"
        cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_distill_extract_sig.sh ──────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

        # Skip si déjà extrait (repartable)
        if [[ -f "$OUT_DIR/sig_embeddings/${TAG}/test.npy" ]]; then
            echo "[skip] $TAG déjà extrait"
            continue
        fi
        echo ""
        echo "─── $TAG (design=$DESIGN teacher=$TEACHER) ───"
        python scripts/context_distill_extract_sig.py \
            --config "$CFG_OVERRIDE" \
            --context-dir "$CONTEXT_DIR" \
            --out-dir "$OUT_DIR" \
            --ckpt-path "$CKPT" \
            --tag "$TAG" \
            $FUSED
        [[ $? -ne 0 ]] && echo "[WARN] $TAG échoué — continuation" >&2
    done
done

echo ""
echo "[slurm] extraction terminée → $OUT_DIR/sig_embeddings/"
ls "$OUT_DIR/sig_embeddings/" 2>/dev/null
