#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Extraction embeddings train/val/test — 6 runs SimDINOv2-B context_distill
# (Design B @512px, LoRA r2a4 ET r8a16, 3 seeds) pour le bootstrap canonique.
#
# POURQUOI CE SCRIPT EXISTE
# -------------------------
# `slurm_context_distill_extract_sig.sh` est câblé sur les 9 runs DINOv3-B
# (R1/R2/R3) ET sur `configs/context_distill_dinov3b.yaml` : il ne peut pas produire
# les runs SimDINOv2-B. Or ces deux configs (0,5057 et 0,5030) sont les rangs 2 et 3
# du tableau maître et sont ABSENTS du palier de significativité (33 groupes, cf.
# `scripts/rapport/significance_tier.py`) — faute d'embeddings. Sans ce job, les
# 2e et 3e meilleurs modèles du projet n'ont ni IC95, ni p, ni BH.
# Vérifié le 2026-09-13 : `$SCRATCH/context_distill/sig_embeddings/` contient bien
# les 9 runs DINOv3-B + les 15 runs du sweep FROZEN, mais AUCUN SimB ctxdistill.
#
# DEUX PRÉ-REQUIS DE CODE (déjà appliqués)
# ----------------------------------------
#   1. `context_distill_extract_sig.py` accepte `--pretrain-checkpoint` et le
#      transmet à `_extract_fused_embeddings`. Sans ça : ValueError "simdinov2_vitb16 :
#      aucun checkpoint fourni" (src/models.py, build_frozen_extractor).
#   2. Le rang LoRA n'a PAS besoin d'être passé : `merge_lora_state_dict` replie
#      l'adaptateur (B·A, scaling lu dans `scaling_buf` du checkpoint) dans les poids
#      de base. r2a4 et r8a16 se chargent donc avec la MÊME config.
#
# Sorties : $SCRATCH/context_distill/sig_embeddings/<tag>/{train,val,test}.npy
#           + *_labels.npy (11 classes, RHOL retirée).
#
# RAPATRIEMENT (depuis le laptop, tunnel MFA ouvert) :
#   rsync -avz --progress \
#     'narval:$SCRATCH/context_distill/sig_embeddings/simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_*' \
#     results/context_distill/sig_embeddings/
# ~470 Mo/run × 6 ≈ 2,8 Go.
#
# Durée estimée : ~40 min/run (forward seul, sans probe) × 6 ≈ 4 h → time 08 h.
# Léger en VRAM (forward seul, MIG a100_3g.20gb).
#
# PRÉ-REQUIS :
#   - $SCRATCH/context_distill/checkpoints/simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_{r2a4,r8a16}_frac100_seed{0,1,2}_best.pth
#   - $SCRATCH/tiles.zip
#   - $SCRATCH/context_512.zip                                       (train)
#   - $SCRATCH/context_512_valtest.zip                               (val/test — Design B)
#   - $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth          (backbone pré-entraîné, REQUIS)
#   - $HOME/benchmark-memoire/vendors/sslplant                       (clone déjà fait par le sweep frozen)
#
# REPARTABLE : chaque run est sauté si `sig_embeddings/<tag>/test.npy` existe déjà.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_simb_extract_sig
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/context_extract_sig_simb_%j.out
#SBATCH --error=logs/context_extract_sig_simb_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE=512
CONFIG="configs/context_distill_simdinov2b.yaml"
PRETRAIN_CKPT="$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
CKPT_DIR="$OUT_DIR/checkpoints"

export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_simb_extract_sig | context=$CONTEXT_SIZE | 6 runs (r2a4/r8a16 × seed 0-2)"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -f "$PRETRAIN_CKPT" ]] && { echo "[ERROR] backbone SimDINOv2-B absent: $PRETRAIN_CKPT"; exit 1; }

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

# Design B : le contexte val/test est requis (fusion à l'extraction), pas seulement train.
VALTEST_ZIP="$SCRATCH/context_${CONTEXT_SIZE}_valtest.zip"
if [[ -f "$VALTEST_ZIP" ]]; then
    echo "[slurm] Design B : merge val/test ($VALTEST_ZIP) ..."
    unzip -q -o "$VALTEST_ZIP" -d "$SLURM_TMPDIR/"
else
    echo "[ERROR] $VALTEST_ZIP introuvable — Design B exige le contexte val/test."; exit 1
fi

CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ -d "$CONTEXT_DIR" ]] || { echo "[ERROR] $CONTEXT_DIR absent"; exit 1; }
echo "[slurm] context_${CONTEXT_SIZE} : $(find "$CONTEXT_DIR" -name '*.png' | wc -l) crops"

# ── 2 configs × 3 seeds ───────────────────────────────────────────────────────
for RANK in r2a4 r8a16; do
    for SEED in 0 1 2; do
        TAG="simdinov2_vitb16_ctxdistill_dB_tSL_ctx${CONTEXT_SIZE}_${RANK}_frac100_seed${SEED}"
        CKPT="$CKPT_DIR/${TAG}_best.pth"
        if [[ ! -f "$CKPT" ]]; then
            echo "[WARN] checkpoint absent : $CKPT — saut" >&2
            continue
        fi

        SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
        [[ -f "$SPATIAL_CSV_DIR/test.csv" ]] || { echo "[WARN] split seed=$SEED absent" >&2; continue; }

        if [[ -f "$OUT_DIR/sig_embeddings/${TAG}/test.npy" ]]; then
            echo "[skip] $TAG déjà extrait"
            continue
        fi

        CFG_OVERRIDE="$SLURM_TMPDIR/cfg_extract_sig_simb_seed${SEED}_${RANK}.yaml"
        cat "$CONFIG" > "$CFG_OVERRIDE"
        cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_extract_sig_simb.sh ─────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

        echo ""
        echo "─── $TAG (Design B @${CONTEXT_SIZE}, LoRA ${RANK}) ───"
        python scripts/context_distill_extract_sig.py \
            --config "$CFG_OVERRIDE" \
            --context-dir "$CONTEXT_DIR" \
            --out-dir "$OUT_DIR" \
            --ckpt-path "$CKPT" \
            --pretrain-checkpoint "$PRETRAIN_CKPT" \
            --tag "$TAG" \
            --fused
        [[ $? -ne 0 ]] && echo "[WARN] $TAG échoué — continuation" >&2
    done
done

echo ""
echo "[slurm] extraction terminée → $OUT_DIR/sig_embeddings/"
ls -d "$OUT_DIR"/sig_embeddings/simdinov2_vitb16_ctxdistill_* 2>/dev/null
echo ""
echo "  Rapatriement :"
echo "    rsync -avz --progress 'narval:\$SCRATCH/context_distill/sig_embeddings/simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_*' \\"
echo "          results/context_distill/sig_embeddings/"
