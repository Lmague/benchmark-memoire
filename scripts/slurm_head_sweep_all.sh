#!/bin/bash
# HEAD-SWEEP COMPLET SUR NARVAL — tout ce que le benchmark a comme embeddings.
#
# Deux phases, un seul job :
#  PHASE 1 — FUSED (24 tags) : $SCRATCH/context_distill/sig_embeddings/
#      têtes : lbfgs, lin-AdamW, MLP-2, bilinéaire-diagonal, FiLM
#      (→ scripts/fusion_head_sweep.py, sorties $SCRATCH/head_results/fusion_heads/)
#  PHASE 2 — TILE-ONLY (33 groupes du palier) : inputs poussés par
#      scripts/push_head_sweep_inputs.sh sous $SCRATCH/head_sweep_inputs/
#      têtes : lbfgs, lin-AdamW, MLP-2 (→ $SCRATCH/head_results/tile_heads/)
#
# Les deux scripts sont idempotents (skip-if-done) : après coupure, re-sbatch.
# Parallellisme : xargs -P 7 processus MONO-THREAD (AGENTS.md §4.8 — jamais de
# BLAS multi-thread, la baseline lbfgs doit reproduire les chiffres canoniques).
#
# Prérequis (locaux, une fois) :
#   python3 scripts/prepare_head_sweep_push.py
#   bash scripts/push_head_sweep_inputs.sh        # ~8-10 Go, rsync incrémental
#
# Soumission : git pull && sbatch scripts/slurm_head_sweep_all.sh
# Budget estimé : phase 1 ≈ 6-9 h + phase 2 ≈ 4-8 h (7 de front) → 48 h allouées.
#
# Rapatriement après fin :
#   scp -r narval:/lustre07/scratch/lmague/head_results/* \
#       results/  (fusion_heads → results/context_distill/, tile_heads → results/rapport_data/)
#
#SBATCH --job-name=head_sweep_all
#SBATCH --account=def-bouguess_gpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/head_sweep_all_%j.out
#SBATCH --error=logs/head_sweep_all_%j.err
set -uo pipefail

cd "$HOME/benchmark-memoire"
git pull --ff-only

VENV="$HOME/ENV/bin/activate"
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV" >&2; exit 1; }
module load python/3.11
source "$VENV"

SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
TILE_ROOT="$SCRATCH/head_sweep_inputs"
OUT_ROOT="$SCRATCH/head_results"
LOGD="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT/fusion_heads" "$OUT_ROOT/tile_heads/logs" "$LOGD" "$LOGD/tile"

echo "═══════ PHASE 1 : FUSED (24 tags) ═══════"
# Garde-fou audit (non bloquant : les tags présents tournent quand même)
python3 scripts/check_fusion_heads_inputs.py --sig-dir "$SIG_DIR" || \
    echo "[warn] audit incomplet — voir rapport ci-dessus" >&2

TAGS=$(ls -d "$SIG_DIR"/*_FROZEN_fused_ctx* "$SIG_DIR"/*_ctxdistill_dB_tL_* \
           "$SIG_DIR"/*_ctxdistill_dA_tL_* "$SIG_DIR"/*_ctxdistill_dA_tEMA_* \
       2>/dev/null | xargs -n1 basename | sort)
echo "$TAGS" | xargs -P 7 -I{} sh -c \
    'python3 -u scripts/fusion_head_sweep.py --sig-dir "$1" --out-dir "$2" --only "$3" \
        > "$4/fused_$3.log" 2>&1 || echo "[ERR fused] $3" >> "$4/ERREURS.log"' \
    _ "$SIG_DIR" "$OUT_ROOT/fusion_heads" {} "$LOGD"
echo "── phase 1 finie : $(ls "$OUT_ROOT/fusion_heads"/*.json 2>/dev/null | wc -l) tags faits"

echo "═══════ PHASE 2 : TILE-ONLY (33 groupes du palier) ═══════"
if [[ ! -d "$TILE_ROOT" ]]; then
    echo "[ERREUR] $TILE_ROOT absent — lancer d'abord (côté local) :"
    echo "  python3 scripts/prepare_head_sweep_push.py && bash scripts/push_head_sweep_inputs.sh"
    exit 1
fi
export TILE_HEAD_OUT="$OUT_ROOT/tile_heads"
GROUPS=$(python3 scripts/tile_head_sweep.py --all --root "$TILE_ROOT")
echo "$GROUPS" | while IFS= read -r g; do
    # 7 de front, noms avec espaces protégés
    while [[ $(jobs -rp | wc -l) -ge 7 ]]; do wait -n; done
    ( python3 -u scripts/tile_head_sweep.py --group "$g" --root "$TILE_ROOT" \
        > "$LOGD/tile/$(echo "$g" | tr ' /()' '____').log" 2>&1 \
        || echo "[ERR tile] $g" >> "$LOGD/ERREURS.log" ) &
done
wait
N=$(ls "$OUT_ROOT/tile_heads"/*.json 2>/dev/null | grep -vc _aggregate)
echo "── phase 2 finie : $N/33 groupes"

# Agrégats finaux
python3 - << 'PY'
import sys, os
sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/rapport")
os.environ.setdefault("TILE_HEAD_OUT", os.path.join(os.environ["SCRATCH"], "head_results/tile_heads"))
import importlib, tile_head_sweep as t
importlib.reload(t); t.aggregate()
print("[agregat] tile_heads/_aggregate.csv régénéré")
PY

echo "═══════ TERMINÉ. Rapatrier : ═══════"
echo "  scp -r narval:\$SCRATCH/head_results/fusion_heads results/context_distill/"
echo "  scp -r narval:\$SCRATCH/head_results/tile_heads   results/rapport_data/"
