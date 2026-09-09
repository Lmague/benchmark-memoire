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

[[ -f /tmp/head_sweep_manifest.txt ]] || { echo "lancer prepare_head_sweep_push.py d'abord"; exit 1; }

rsync -av --relative --files-from=/tmp/head_sweep_manifest.txt \
    . "$NARVAL:/lustre07/scratch/lmague/head_sweep_inputs/"

echo "[push] terminé — inputs sous \$SCRATCH/head_sweep_inputs/ (arborescence miroir du dépôt)"
