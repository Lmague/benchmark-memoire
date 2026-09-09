#!/usr/bin/env bash
# Vérifie AVANT tout upload ce qui existe déjà sur Narval parmi les inputs du
# head-sweep. Trois temps :
#   1. inventaire remote (find de tous les .npy de $SCRATCH + dépôt cloné)
#   2. reconcile local (fingerprint md5 64 Ko tête+queue : nom+taille ne
#      suffisent pas, tous les runs ont un train.npy de la même taille…)
#   3. ce qui existe déjà → lien symbolique sur place (0 octet transporté) ;
#      ce qui manque → manifeste rsync résiduel.
#
# Usage local (MFA ssh) :  bash scripts/check_narval_inputs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NARVAL=${NARVAL:-narval}

[[ -f /tmp/head_sweep_manifest.txt ]] || python3 scripts/prepare_head_sweep_push.py

echo "[check] inventaire .npy sur Narval (~1-2 min, Lustre)…"
ssh "$NARVAL" 'find /lustre07/scratch/lmague /home/lmague/benchmark-memoire \
    -name "*.npy" -printf "%p|%s\n" 2>/dev/null' > /tmp/narval_npy.txt
echo "[check] $(wc -l < /tmp/narval_npy.txt) .npy trouvés"

python3 scripts/head_sweep_reconcile.py "$NARVAL"

echo
echo "══════════ SUITE (valides-tu le rapport ci-dessus d'abord) ══════════"
echo "# a) créer les liens sur place :"
echo "ssh $NARVAL 'bash -s' < /tmp/head_sweep_link.sh"
echo "# b) pousser uniquement les fichiers vraiment absents :"
echo "MANIFEST=/tmp/head_sweep_missing.txt bash scripts/push_head_sweep_inputs.sh"
