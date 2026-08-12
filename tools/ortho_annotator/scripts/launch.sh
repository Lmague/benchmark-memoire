#!/usr/bin/env bash
# Lanceur "app" de l'annotateur : choisit l'orthomosaïque de départ (mémorise le
# dernier choix), démarre le serveur si besoin, ouvre le navigateur.
# Prévu pour être appelé depuis un lanceur d'applications (entrée .desktop),
# mais fonctionne aussi en ligne de commande.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOOL_ROOT="$REPO_ROOT/tools/ortho_annotator"
PYTHON="/home/erazal/miniconda3/bin/python"
ORTHO_DIR="$REPO_ROOT/Dataset_Leo/Orthomosaiques"
OUTPUT="$HOME/annotations_leo/session.gpkg"
PORT=8000

STATE_DIR="$HOME/.local/state/ortho-annotator"
STATE_FILE="$STATE_DIR/last_raster"
LOG_FILE="$STATE_DIR/server.log"
mkdir -p "$STATE_DIR" "$(dirname "$OUTPUT")"

notify() {
    if command -v notify-send >/dev/null 2>&1; then
        notify-send -a "Ortho Annotator" "$1" "$2"
    fi
}

fail() {
    notify "Échec du lancement" "$1"
    zenity --error --title="Ortho Annotator" --text="$1" 2>/dev/null || true
    exit 1
}

DEFAULT_RASTER=""
[ -f "$STATE_FILE" ] && DEFAULT_RASTER="$(cat "$STATE_FILE")"
[ -f "$DEFAULT_RASTER" ] || DEFAULT_RASTER="$ORTHO_DIR/Orthom_Clairiere_9Aout23_WGS84UTM18N.tif"

server_up() {
    curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/"
}

if ! server_up; then
    RASTER="$(zenity --file-selection \
        --title="Ortho Annotator : choisir l'orthomosaïque de départ" \
        --filename="$DEFAULT_RASTER" \
        --file-filter="Orthomosaïques (*.tif) | *.tif" 2>/dev/null)" || exit 0
    [ -f "$RASTER" ] || fail "Fichier introuvable : $RASTER"
    echo "$RASTER" > "$STATE_FILE"

    cd "$TOOL_ROOT"
    nohup "$PYTHON" -m ortho_annotator serve \
        --raster "$RASTER" --output "$OUTPUT" \
        >"$LOG_FILE" 2>&1 &
    disown

    for _ in $(seq 1 120); do
        server_up && break
        sleep 0.5
    done
    if ! server_up; then
        fail "Le serveur n'a pas démarré. Voir $LOG_FILE"$'\n\n'"$(tail -n 20 "$LOG_FILE")"
    fi
fi

xdg-open "http://127.0.0.1:$PORT/" >/dev/null 2>&1 &
notify "Annotateur prêt" "http://127.0.0.1:$PORT/"
