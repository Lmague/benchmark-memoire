#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Contrôles de la fusion tuile⊕contexte (Design B) — attribuer le gain de R2.
#
# Calcule la sonde canonique sur features fusionnées (1536) ET tuile seule (768)
# pour deux modèles SANS distillation, sur les 3 splits spatiaux frac100 :
#   - DINOv3-B GELÉ (contrôle 1) : le contexte aide-t-il même sans entraînement ?
#   - LoRA spatial de base r8a16 (contrôle 2, baseline canonique ~0.4827) :
#     l'entraînement LoRA seul suffit-il à utiliser le contexte ?
#
# Comparaison cible (mêmes splits, même sonde, même métrique) :
#   ctxdistill Design B fused = 0.5080 ± 0.0016   (R2, à attribuer)
#   ctxdistill Design A tile  = 0.4870 ± 0.0013   (R1)
#   LoRA spatial r8a16 tile   ≈ 0.4827 ± 0.0049   (CANONICAL csv)
#
# Attribution :
#   frozen/fused ≈ 0.508 → le gain vient du contexte dans la sonde, pas de
#                          l'entraînement (le 0.508 est une borne « info »).
#   frozen/fused ≈ 0.487 → c'est l'entraînement qui débloque le contexte.
#
# Sorties : $SCRATCH/context_distill/controls/fused_probe_{frozen|lora_base_r8a16}_seed{S}.json
#
# PRÉ-REQUIS (déjà en place sur Narval) :
#   - $SCRATCH/tiles.zip, $SCRATCH/context_1024.zip (val/test inclus)
#   - $SCRATCH/sota_screening/lora_spatial_v2/frac100/checkpoints/
#       dinov3_vitb16_lvd_lora_r8a16_frac100_seed{0,1,2}_best.pth   (contrôle 2)
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/ (frozen)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_fused_control
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/context_fused_probe_controls_%j.out
#SBATCH --error=logs/context_fused_probe_controls_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE="${1:-1024}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
BASE_CKPT_DIR="$SCRATCH/sota_screening/lora_spatial_v2/frac100/checkpoints"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_fused_control | context=$CONTEXT_SIZE | frozen + lora_base_r8a16"
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

for SEED in 0 1 2; do
    SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
    if [[ ! -f "$SPATIAL_CSV_DIR/test.csv" ]]; then
        echo "[WARN] split spatial absent pour seed=$SEED — sauté" >&2
        continue
    fi

    CFG_OVERRIDE="$SLURM_TMPDIR/cfg_fused_control_seed${SEED}.yaml"
    cat "$CONFIG" > "$CFG_OVERRIDE"
    cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_fused_probe_controls.sh ─────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

    echo ""
    echo "─── seed=$SEED : contrôle 1 = DINOv3-B GELÉ ───"
    python scripts/context_fused_probe_controls.py \
        --config "$CFG_OVERRIDE" \
        --context-dir "$CONTEXT_DIR" \
        --out-dir "$OUT_DIR" \
        --seed "$SEED" \
        --tag frozen --frozen
    [[ $? -ne 0 ]] && echo "[WARN] contrôle frozen seed=$SEED échoué" >&2

    BASE_CKPT="$BASE_CKPT_DIR/dinov3_vitb16_lvd_lora_r8a16_frac100_seed${SEED}_best.pth"
    echo ""
    echo "─── seed=$SEED : contrôle 2 = LoRA spatial r8a16 ($BASE_CKPT) ───"
    if [[ -f "$BASE_CKPT" ]]; then
        python scripts/context_fused_probe_controls.py \
            --config "$CFG_OVERRIDE" \
            --context-dir "$CONTEXT_DIR" \
            --out-dir "$OUT_DIR" \
            --seed "$SEED" \
            --tag lora_base_r8a16 \
            --ckpt-path "$BASE_CKPT"
        [[ $? -ne 0 ]] && echo "[WARN] contrôle lora_base seed=$SEED échoué" >&2
    else
        echo "[WARN] checkpoint baseline absent : $BASE_CKPT — contrôle 2 sauté pour seed=$SEED" >&2
    fi
done

echo ""
echo "[slurm] contrôles terminés → $OUT_DIR/controls/"
ls -la "$OUT_DIR/controls/" 2>/dev/null
