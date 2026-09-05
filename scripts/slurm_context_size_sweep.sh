#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP DES TAILLES DE CONTEXTE — UN SEUL JOB qui fait TOUT (extraction + probes).
#
# Demande de M. Bouguessa : « regarder les différentes tailles de contexte (512,
# 1024, 2048), avant de passer à des modèles DINOv3 plus grands ».
#
# Ce job fait, dans un SEUL sbatch, pour chacune des tailles :
#   1. EXTRACTION GPU des features FROZEN fusionnées [tile ; ctx] (3 splits) →
#      $SCRATCH/context_distill/sig_embeddings/<frozen_ctx<size>>/
#      (script : scripts/context_size_sweep.py --skip-probes)
#   2. PROBES CPU canoniques (sonde lbfgs, BLAS mono-thread) : fused / tile /
#      ctx-seul / contexte-permuté×3
#      (script : scripts/context_bouguessa_controls.py)
#
# Lecture (ce que ça « dit ») — voir l'en-tête de context_bouguessa_controls.py :
#   - courbe d'information  = fused(512/1024/2048) − tile
#   - effet de concat seule = fused_ctxperm − tile
#   - effet spatial (vrai)  = fused − fused_ctxperm
#     · fused_ctxperm ≈ tile  → le gain vient bien de l'appariement spatial ;
#     · fused_ctxperm ≈ fused → c'est seulement l'effet d'ensemble du concat.
#
# Le pallier ENTRAÎNÉ (R2 Design B à 512/2048) est un job à part, lancé après si la
# courbe frozen le justifie : sbatch scripts/slurm_context_distill.sh <size> B.
#
# REPARTABLE : chaque extraction est sautée si les sig_embeddings existent, chaque
# sonde est sautée si son JSON existe → resoumettre redémarre où ça en était.
#
# Sorties :
#   - $SCRATCH/context_distill/sig_embeddings/dinov3_vitb16_lvd_FROZEN_fused_ctx<size>_frac100_seed0/
#   - $SCRATCH/context_distill/controls_bouguessa/frozen_ctx<size>_seed0_*.json
# Rapatriement : scp -r narval:$SCRATCH/context_distill/controls_bouguessa results/context_distill/
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip
#   - $SCRATCH/context_512.zip, context_1024.zip, context_2048.zip  (train crops ;
#     1024 contient déjà val/test) ET, pour 512/2048, les val/test manquants :
#     $SCRATCH/context_512_valtest.zip + context_2048_valtest.zip
#     (produits EN LOCAL par scripts/context_crop.py sur val.csv+test.csv —
#     cf. NEEDS_HUMAN.md). Les deux zips sont fusionnés dans le même dossier
#     (aucune collision de noms : tuiles disjointes).
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#
# Soumission : git pull && sbatch scripts/slurm_context_size_sweep.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_size_sweep
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/context_size_sweep_%j.out
#SBATCH --error=logs/context_size_sweep_%j.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
OUT_DIR="$SCRATCH/context_distill"
SIZES="${1:-512 1024 2048}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_size_sweep (UN job : extraction + probes) | tailles : $SIZES"
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

for SIZE in $SIZES; do
    echo ""
    echo "═══════ taille de contexte : ${SIZE}px ═══════"
    CONTEXT_DIR="$SLURM_TMPDIR/context_${SIZE}"
    rm -rf "$CONTEXT_DIR"
    if [[ ! -f "$SCRATCH/context_${SIZE}.zip" ]]; then
        echo "[ERROR] $SCRATCH/context_${SIZE}.zip introuvable — taille $SIZE sautée" >&2
        continue
    fi
    unzip -q "$SCRATCH/context_${SIZE}.zip" -d "$SLURM_TMPDIR/"
    # val/test manquants pour 512/2048 : merge du zip complémentaire (tuiles disjointes).
    if [[ -f "$SCRATCH/context_${SIZE}_valtest.zip" ]]; then
        echo "[slurm] Merge val/test (${SIZE}px) ..."
        unzip -q "$SCRATCH/context_${SIZE}_valtest.zip" -d "$SLURM_TMPDIR/"
    fi
    N_TILES=$(find "$CONTEXT_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo "[slurm] context_${SIZE} : $N_TILES crops"
    if [[ "$N_TILES" -lt 79000 ]]; then
        echo "[WARN] ${SIZE}px : $N_TILES crops < 79k attendus (49433 train + 13209 val + 17598 test) — vérifier les zips" >&2
    fi

    echo "─── [1/2] extraction frozen fusion −${SIZE}px ───"
    SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed0"
    if [[ ! -f "$SPATIAL_CSV_DIR/test.csv" ]]; then
        echo "[ERROR] split spatial absent : $SPATIAL_CSV_DIR/test.csv (git pull ?)" >&2
        continue
    fi
    CFG_OVERRIDE="$SLURM_TMPDIR/cfg_size_sweep_${SIZE}.yaml"
    cat "$CONFIG" > "$CFG_OVERRIDE"
    cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_size_sweep.sh ───────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF
    python scripts/context_size_sweep.py \
        --config "$CFG_OVERRIDE" \
        --context-dir "$CONTEXT_DIR" \
        --context-size "$SIZE" \
        --out-dir "$OUT_DIR" \
        --skip-probes
    [[ $? -ne 0 ]] && echo "[WARN] extraction ${SIZE}px échouée" >&2

    echo "─── [2/2] probes CPU −${SIZE}px (fused/tile/ctx/perm×3) ───"
    # ⚠️ Contrairement à une impression naturelle, le tag SONDE est SANS suffixe
    # _seed0 : context_bouguessa_controls._load_split ajoute déjà "_seed{seed}" au
    # tag (convention R2, où le tag par défaut est sans seed). Le dossier
    # d'extraction, lui, est nommé _sig_tag() = "…_frac100_seed0". Donc :
    #   sonde lit  ${SIG_DIR}/${TAG}_seed0/test.npy
    #   =          ${SIG_DIR}/…_FROZEN_fused_ctx${SIZE}_frac100_seed0/test.npy  ✓
    TAG="dinov3_vitb16_lvd_FROZEN_fused_ctx${SIZE}_frac100"
    if [[ ! -f "$SIG_DIR/${TAG}_seed0/test.npy" ]]; then
        echo "[WARN] $SIG_DIR/${TAG}_seed0/test.npy absent — probes ${SIZE}px sautés (extraction échouée ?)" >&2
        continue
    fi
    python scripts/context_bouguessa_controls.py \
        --sig-dir "$SIG_DIR" \
        --tag "$TAG" \
        --seeds 0 \
        --perm-reps 3 \
        --json-prefix "frozen_ctx${SIZE}" \
        --out-dir "$OUT_DIR" \
        --workers 4
    [[ $? -ne 0 ]] && echo "[WARN] probes ${SIZE}px échoués" >&2
done

echo ""
echo "[slurm] terminé → $OUT_DIR/controls_bouguessa/"
ls -la "$OUT_DIR/controls_bouguessa/" 2>/dev/null || true
echo ""
echo "[slurm] Rapatriement (depuis la machine locale) :"
echo "  scp -r narval:\$SCRATCH/context_distill/controls_bouguessa results/context_distill/"
