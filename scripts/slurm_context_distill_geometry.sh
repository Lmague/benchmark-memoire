#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Géométrie de l'espace latent des modèles context_distill (R1/R2/R3, ctx1024).
#
# Recharge les 9 checkpoints *_best.pth déjà produits (jobs 2090475/2090476/2090477)
# et calcule RankMe / anisotropie / α-ReQ / NESum sur le split test (convention
# canonique du dépôt : rankme_split=test, subsample 20k, seed 42 — même pipeline
# que analyze.py via src.latent). Design B : géométrie sur la représentation
# fusionnée (1536) ET la branche tuile seule (768) du même checkpoint.
#
# Léger : 9 checkpoints × ~17,6k images, forward SEUL sur GPU → quelques minutes.
# Ne ré-entraîne RIEN, lecture seule des runs existants. À relancer après coup sur
# les runs terminés — aucun ordre imposé avec les jobs d'entraînement.
#
# Sorties : $SCRATCH/context_distill/geometry/geometry_seed{S}_splittest.json
#           + embeddings .npy (fp16) sous geometry/embeddings/ → à rapatrier en
#           local (petits fichiers, ~quelques centaines de Mo au total) pour toute
#           analyse ultérieure sans GPU.
#
# PRÉ-REQUIS (déjà en place sur Narval) :
#   - $SCRATCH/context_distill/checkpoints/*_best.pth        (les 9 runs)
#   - $SCRATCH/tiles.zip                                     (tuiles 224px)
#   - $SCRATCH/context_1024.zip                               (contexte, val/test inclus pour B)
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#
# Usage : $1 = context_size (défaut 1024). $2 = split (défaut test).
#   sbatch scripts/slurm_context_distill_geometry.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctxdistill_geometry
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/context_distill_geometry_%j.out
#SBATCH --error=logs/context_distill_geometry_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE="${1:-1024}"
SPLIT="${2:-test}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctxdistill_geometry | context=$CONTEXT_SIZE split=$SPLIT"
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
    echo "[ERROR] $CONTEXT_ZIP introuvable (Design B en dépend)" >&2
    exit 1
fi
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ ! -d "$CONTEXT_DIR" ]] && { echo "[ERROR] $CONTEXT_DIR absent après unzip"; exit 1; }

# 3 seeds : le split de géométrie est celui du run correspondant (spatial frac100).
for SEED in 0 1 2; do
    SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
    if [[ ! -f "$SPATIAL_CSV_DIR/test.csv" ]]; then
        echo "[WARN] split spatial absent pour seed=$SEED — sauté" >&2
        continue
    fi

    CFG_OVERRIDE="$SLURM_TMPDIR/cfg_geometry_seed${SEED}.yaml"
    cat "$CONFIG" > "$CFG_OVERRIDE"
    cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_distill_geometry.sh ──────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

    echo ""
    echo "─── seed=$SEED ───"
    python scripts/context_distill_geometry.py \
        --config "$CFG_OVERRIDE" \
        --context-dir "$CONTEXT_DIR" \
        --out-dir "$OUT_DIR" \
        --seed "$SEED" \
        --split "$SPLIT" \
        --save-embeddings

    if [[ $? -ne 0 ]]; then
        echo "[WARN] seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] géométrie terminée → $OUT_DIR/geometry/"
ls -la "$OUT_DIR/geometry/" 2>/dev/null
