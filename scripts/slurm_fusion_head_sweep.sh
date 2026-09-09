#!/bin/bash
# SWEEP FUSION-HEADS NON LINÉAIRES — sur TOUS les embeddings fused de Narval.
#
# Que fait-il : scripts/fusion_head_sweep.py sonde chaque dossier
#   $SCRATCH/context_distill/sig_embeddings/<tag>/{train,val,test}.npy
# avec 5 têtes : lbfgs canonique (baseline mono-thread, AGENTS.md §4.8),
# linéaire AdamW (contrôle d'optimiseur), MLP-2 couches, bilinéaire diagonal,
# FiLM. Têtes bilinéaire/FiLM uniquement sur les tags fusionnés.
#
# Tags couverts (mêmes défauts que le python) :
#   *_FROZEN_fused_ctx*   : sweep gelé 5 backbones × {512,1024,2048} (seed 0)
#   *_ctxdistill_dB_tL_*  : R2 entraîné, 3 seeds (1536)
#   *_ctxdistill_dA_*     : R1/R3 (768, vue seule — sanity)
#
# PARALLÉLISME : un processus MONO-THREAD par tag (xargs -P), jamais de BLAS
# multi-thread (la baseline lbfgs doit reproduire les chiffres canoniques).
# Idempotent : un <tag>.json existant est sauté → re-sbatch après coupure.
#
# Sorties (sur Narval) :
#   $SCRATCH/context_distill/fusion_heads/<tag>.json
#   $SCRATCH/context_distill/fusion_heads/_aggregate.csv (régénéré par le
#     dernier processus, ou à relancer seul avec --only d'un tag déjà fait)
# Rapatriement local APRÈS la fin du job :
#   scp -r narval:/lustre07/scratch/lmague/context_distill/fusion_heads \
#       results/context_distill/
#
# Budget estimé : ~24 tags × 1-3 h / 7 de front → 8-15 h wall. 36 h demandées
# = marge x2 ; si coupure, re-sbatch (skip-if-done).
#
# Soumission :  git pull && sbatch scripts/slurm_fusion_head_sweep.sh
#
#SBATCH --job-name=fusion_heads
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/fusion_heads_%j.out
#SBATCH --error=logs/fusion_heads_%j.err
set -euo pipefail

cd "$HOME/benchmark-memoire"
git pull --ff-only

VENV="$HOME/ENV/bin/activate"
[[ -f "$VENV" ]] && { module load python/3.11; source "$VENV"; } \
    || echo "[WARN] venv absent — numpy/torch introuvables ?"

SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
OUT_DIR="$SCRATCH/context_distill/fusion_heads"

if [[ ! -d "$SIG_DIR" ]]; then
    echo "[ERROR] $SIG_DIR introuvable" >&2; exit 1
fi
mkdir -p "$OUT_DIR" logs

# GARDE-FOU : audit des inputs avant de consommer l'allocation.
# SKIP_AUDIT=1 sbatch ... pour forcer si l'audit est déjà passé ailleurs.
if [[ "${SKIP_AUDIT:-0}" != "1" ]]; then
    if ! python3 scripts/check_fusion_heads_inputs.py --sig-dir "$SIG_DIR"; then
        echo "[FUSION-HEADS] audit EN ÉCHEC — le sweep tournerait partiel." >&2
        echo "[FUSION-HEADS] relancer avec SKIP_AUDIT=1 sbatch $0 pour forcer," >&2
        echo "[FUSION-HEADS] ou ré-extraire les tags manquants d'abord." >&2
        exit 1
    fi
fi
# Liste des tags (mêmes motifs que le python par défaut)
TAGS=$(ls -d \
    "$SIG_DIR"/*_FROZEN_fused_ctx* \
    "$SIG_DIR"/*_ctxdistill_dB_tL_* \
    "$SIG_DIR"/*_ctxdistill_dA_tL_* \
    "$SIG_DIR"/*_ctxdistill_dA_tEMA_* \
    2>/dev/null | xargs -n1 basename | sort)

if [[ -z "$TAGS" ]]; then
    echo "[ERROR] aucun tag ne correspond dans $SIG_DIR" >&2; exit 1
fi
echo "[fusion-heads] $(echo "$TAGS" | wc -l) tags, 7 processus mono-thread de front"

# xargs : un process python par tag (mono-thread forcé dans le python).
# Les tags déjà faits sont relancés mais sautés en 2 s (skip-if-done).
LOGD="$SCRATCH/context_distill/logs_fusion"
mkdir -p "$LOGD"
echo "$TAGS" | xargs -P 7 -I{} sh -c \
    'python3 scripts/fusion_head_sweep.py --sig-dir "$1" --out-dir "$2" --only "$3" > "$4/$3.log" 2>&1 || echo "ERREUR tag $3 (voir $4/$3.log)"' \
    _ "$SIG_DIR" "$OUT_DIR" {} "$LOGD"

# Agrégat final : un dernier run sur un tag existant régénère le CSV complet.
LAST=$(echo "$TAGS" | head -1)
python3 scripts/fusion_head_sweep.py --sig-dir "$SIG_DIR" --out-dir "$OUT_DIR" \
    --only "$LAST" >/dev/null 2>&1 || true

echo "[fusion-heads] TERMINÉ. Rapatrier :"
echo "  scp -r narval:$OUT_DIR results/context_distill/"
