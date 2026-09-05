#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP CONTEXTE MULTI-BACKBONES (frozen, SANS entraînement) — 3 tailles × N modèles.
#
# Répond à « regarder les tailles de contexte (512/1024/2048) avant de passer à des
# modèles DINOv3 plus grands » (Bouguessa) ET à l'extension « est-ce que le gain
# contexte tient à d'autres échelles ? » : courbes 512→1024→2048 pour DINOv3-S,
# DINOv3-L, SimDINOv2-B, SimDINOv2-L — en FROZEN (aucun entraînement, ~1-3 h GPU
# selon le modèle). On décide ensuite quoi entraîner sur la base des courbes.
#
# Pour chaque modèle × taille : extraction GPU des features frozen FUSIONNÉES
# [tile ; ctx] (train/val/test, skip-if-done) puis probes CPU (fused/tile/ctx/perm×3,
# skip-if-json). Même machinerie que le sweep DINOv3-B (context_size_sweep.py +
# context_bouguessa_controls.py), tag = <model_tag>_FROZEN_fused_ctx<size>_frac100_seed0.
#
# MODÈLES (par défaut) : spec = "model_tag|config|chemin_checkpoint(optionnel)"
#   - dinov3_vits16|frozen_dinov3_vits16.yaml              (HF, dim 384, ~22 M)
#   - dinov3_vitl16|frozen_dinov3_lvd_large.yaml           (HF, dim 1024, ~300 M)
#   - simdinov2_vitb16|frozen_simdinov2_vitb16.yaml|$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth
#   - simdinov2_vitl16|frozen_simdinov2_vitl16.yaml|$SCRATCH/checkpoints/simdinov2_vitl_inat21plantae.pth
#
# Sorties : $SCRATCH/context_distill/sig_embeddings/<tag>/
#           $SCRATCH/context_distill/controls_bouguessa/frozen_<model>_ctx<size>_seed0_*.json
#
# PRÉ-REQUIS : zips contexte sur $SCRATCH (context_{512,1024,2048}.zip incl. val/test
# via context_*_valtest.zip mergés par le job) + $SCRATCH/checkpoints/simdinov2_*.pth
# + clés HF (DINOv3) dans $SCRATCH/hf_cache.
#
# Usage : sbatch scripts/slurm_context_frozen_models.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_frozen_models
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/context_frozen_models_%j.out
#SBATCH --error=logs/context_frozen_models_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
OUT_DIR="$SCRATCH/context_distill"
SIZES="${2:-512 1024 2048}"

# spec par défaut : "model_tag|config_yaml|checkpoint(optionnel)"
DEFAULT_MODELS="dinov3_vits16|frozen_dinov3_vits16.yaml|
dinov3_vitl16|frozen_dinov3_lvd_large.yaml|
simdinov2_vitb16|frozen_simdinov2_vitb16.yaml|${SCRATCH}/checkpoints/simdinov2_vitb_inat21plantae.pth
simdinov2_vitl16|frozen_simdinov2_vitl16.yaml|${SCRATCH}/checkpoints/simdinov2_vitl_inat21plantae.pth"
MODELS="${1:-$DEFAULT_MODELS}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_frozen_models | tailles : $SIZES"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR (git pull ?)"; exit 1; }
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

SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed0"
[[ ! -f "$SPATIAL_CSV_DIR/test.csv" ]] && { echo "[ERROR] split spatial absent : $SPATIAL_CSV_DIR"; exit 1; }

while IFS='|' read -r MODEL_TAG CONFIG_YAML CKPT; do
    [[ -z "$MODEL_TAG" ]] && continue
    echo ""
    echo "═══════════ MODÈLE : $MODEL_TAG (config: $CONFIG_YAML, ckpt: ${CKPT:-HF}) ═══════════"
    [[ ! -f "configs/$CONFIG_YAML" ]] && { echo "[ERROR] config introuvable : configs/$CONFIG_YAML — modèle sauté" >&2; continue; }
    if [[ -n "$CKPT" && ! -f "$CKPT" ]]; then
        echo "[ERROR] checkpoint absent : $CKPT — modèle $MODEL_TAG sauté" >&2
        continue
    fi

    for SIZE in $SIZES; do
        echo ""
        echo "───── ${MODEL_TAG} · contexte ${SIZE}px ─────"
        CONTEXT_DIR="$SLURM_TMPDIR/context_${SIZE}"
        rm -rf "$CONTEXT_DIR"
        if [[ ! -f "$SCRATCH/context_${SIZE}.zip" ]]; then
            echo "[ERROR] $SCRATCH/context_${SIZE}.zip introuvable — taille $SIZE sautée" >&2
            continue
        fi
        unzip -q "$SCRATCH/context_${SIZE}.zip" -d "$SLURM_TMPDIR/"
        if [[ -f "$SCRATCH/context_${SIZE}_valtest.zip" ]]; then
            echo "[slurm] Merge val/test (${SIZE}px) ..."
            unzip -q "$SCRATCH/context_${SIZE}_valtest.zip" -d "$SLURM_TMPDIR/"
        fi
        N_TILES=$(find "$CONTEXT_DIR" -name "*.png" 2>/dev/null | wc -l)
        echo "[slurm] context_${SIZE} : $N_TILES crops"
        if [[ "$N_TILES" -lt 79000 ]]; then
            echo "[WARN] ${SIZE}px : $N_TILES crops < 79k attendus — vérifier les zips" >&2
        fi

        CFG_OVERRIDE="$SLURM_TMPDIR/cfg_frozen_${MODEL_TAG}_${SIZE}.yaml"
        cat "configs/$CONFIG_YAML" > "$CFG_OVERRIDE"
        cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_frozen_models.sh ──────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

        echo "[1/2] extraction frozen fusion −${MODEL_TAG} ${SIZE}px"
        CKPT_ARG=()
        [[ -n "$CKPT" ]] && CKPT_ARG=(--checkpoint "$CKPT")
        python scripts/context_size_sweep.py \
            --config "$CFG_OVERRIDE" \
            --context-dir "$CONTEXT_DIR" \
            --context-size "$SIZE" \
            --model-tag "$MODEL_TAG" \
            "${CKPT_ARG[@]}" \
            --out-dir "$OUT_DIR" \
            --skip-probes
        [[ $? -ne 0 ]] && echo "[WARN] extraction ${MODEL_TAG} ${SIZE}px échouée" >&2

        echo "[2/2] probes CPU −${MODEL_TAG} ${SIZE}px (fused/tile/ctx/perm×3)"
        TAG="${MODEL_TAG}_FROZEN_fused_ctx${SIZE}_frac100"
        if [[ ! -f "$SIG_DIR/${TAG}_seed0/test.npy" ]]; then
            echo "[WARN] $SIG_DIR/${TAG}_seed0/test.npy absent — probes ${MODEL_TAG} ${SIZE}px sautés" >&2
            continue
        fi
        python scripts/context_bouguessa_controls.py \
            --sig-dir "$SIG_DIR" \
            --tag "$TAG" \
            --seeds 0 \
            --perm-reps 3 \
            --json-prefix "frozen_${MODEL_TAG}_ctx${SIZE}" \
            --out-dir "$OUT_DIR" \
            --workers 4
        [[ $? -ne 0 ]] && echo "[WARN] probes ${MODEL_TAG} ${SIZE}px échoués" >&2
    done
done <<< "$MODELS"

echo ""
echo "[slurm] terminé → $OUT_DIR/controls_bouguessa/"
ls -la "$OUT_DIR/controls_bouguessa/" 2>/dev/null | tail -30
echo ""
echo "[slurm] Rapatriement (depuis la machine locale) :"
echo "  rsync -avz --progress lmague@narval.alliancecan.ca:/scratch/lmague/context_distill/controls_bouguessa/ results/context_distill/controls_bouguessa/"