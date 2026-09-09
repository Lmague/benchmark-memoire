#!/usr/bin/env bash
# Push des inputs du head-sweep vers Narval.
#
# Prérequis : python3 scripts/prepare_head_sweep_push.py
# Lancer ICI (MFA ssh requis) :  bash scripts/push_head_sweep_inputs.sh
# Idempotent : rsync ne renvoie que les nouveautés. ~8-10 Go la première fois.
#
# Arrivée sur Narval : $SCRATCH/head_sweep_inputs/<mêmes chemins relatifs que
# le dépôt> — c'est ce que lit `tile_head_sweep.py --root
# $SCRATCH/head_sweep_inputs` dans slurm_head_sweep_all.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

NARVAL=${NARVAL:-narval}
# MANIFEST=/tmp/head_sweep_missing.txt pour ne pousser que ce que Narval n'a
# pas déjà (voir scripts/check_narval_inputs.sh — à lancer AVANT ce push).
MANIFEST=${MANIFEST:-/tmp/head_sweep_manifest.txt}

[[ -f "$MANIFEST" ]] || { echo "lancer prepare_head_sweep_push.py (ou check_narval_inputs.sh) d'abord"; exit 1; }

echo "[push] $(wc -l < "$MANIFEST") fichiers depuis $MANIFEST"
rsync -av --relative --files-from="$MANIFEST" \
    . "$NARVAL:/lustre07/scratch/lmague/head_sweep_inputs/"

echo "[push] terminé — inputs sous \$SCRATCH/head_sweep_inputs/ (arborescence miroir du dépôt)"
