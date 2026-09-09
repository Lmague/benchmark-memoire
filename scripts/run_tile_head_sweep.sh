#!/usr/bin/env bash
# Lança le tile_head_sweep sur les 33 groupes du palier, en parallèle par
# PROCESSUS (un par groupe, mono-thread BLAS — AGENTS.md §4.8). Idempotent :
# un JSON de groupe existant est sauté → relancer après coupure reprend.
#
# Logs : results/rapport_data/tile_heads/logs/<slug>.log
# Fin  : rapport de synthèse (OK/ERROR/skip) dans tile_heads/_runner.log
#
# Lancement détaché recommandé :
#   nohup bash scripts/run_tile_head_sweep.sh > /dev/null 2>&1 &
# Surveillance :
#   tail -f results/rapport_data/tile_heads/_runner.log
#   ls results/rapport_data/tile_heads/*.json | wc -l   # cible : 33
set -uo pipefail
cd "$(dirname "$0")/.."

OUTDIR=results/rapport_data/tile_heads
mkdir -p "$OUTDIR/logs"
NPROC=${NPROC:-7}

echo "[runner] $(date) — $(python3 scripts/tile_head_sweep.py --all | wc -l) groupes, $NPROC de front" \
    | tee "$OUTDIR/_runner.log"

python3 scripts/tile_head_sweep.py --all | xargs -d '\n' -P "$NPROC" -I{} \
    sh -c 'python3 -u scripts/tile_head_sweep.py --group "$1" > "'"$OUTDIR"'/logs/$(echo "$1" | tr " ()/" "____").log" 2>&1 || echo "[runner] ERREUR: $1" | tee -a "'"$OUTDIR"'/_runner.log"' _ {}

N=$(ls "$OUTDIR"/*.json 2>/dev/null | grep -v _aggregate | wc -l)
echo "[runner] $(date) — TERMINÉ : $N/33 JSON produits. Agrégat : $OUTDIR/_aggregate.csv" \
    | tee -a "$OUTDIR/_runner.log"
