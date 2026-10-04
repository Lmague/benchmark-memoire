#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Encode les annotations Léo (bank + négatifs) avec PLUSIEURS modèles, pour la
# comparaison appariée : SimDINOv2-B iNat gelé (baseline) vs les 3 seeds ExPLoRA-Léo.
#
# Tourne EN LOCAL (le raster COG n'est pas sur Narval) : ~10-15 min par modèle pour
# la phase `bank` (6000 points), ~10-20 min pour `negatives`. La phase `grid`
# (66 816 tuiles) coûte ~1,5-2 h par modèle → à ne lancer que pour l'outil d'annotation.
#
# Usage :
#   bash scripts/leo_ssl_annot_extract.sh                        # bank seulement, 4 modèles
#   PHASES=bank,negatives bash scripts/leo_ssl_annot_extract.sh
#   PHASES=bank,negatives,grid bash scripts/leo_ssl_annot_extract.sh
#   TAGS="base:/chemin/a.pth ssl_seed0:/chemin/b.pth" bash scripts/leo_ssl_annot_extract.sh
#
# Les checkpoints ExPLoRA-Léo se rapatrient depuis Narval (1 par seed suffit pour
# commencer ; `last.pth` est le plus long entraîné) :
#   rsync -avP narval:$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed0/checkpoints/last.pth \
#       ~/Documents/Mémoire/checkpoints/leo_ssl/seed0_last.pth
# ═══════════════════════════════════════════════════════════════════════════════
set -u

REPO="${REPO:-$HOME/Documents/Mémoire}"
ANNOT="${ANNOT:-$HOME/annotations_leo}"
EMB="${EMB:-$ANNOT/embeddings}"
PHASES="${PHASES:-bank}"
PY="${PY:-/home/erazal/miniconda3/bin/python}"
THREADS="${LEO_THREADS:-16}"
INAT="$REPO/checkpoints/simdinov2_vitb_inat21plantae.pth"
SSL_DIR="${SSL_DIR:-$REPO/checkpoints/leo_ssl}"

# TAGS : liste "tag:chemin_ckpt". Défaut = baseline + 3 seeds si les .pth sont là.
if [[ -n "${TAGS:-}" ]]; then
    read -ra SPECS <<< "$TAGS"
else
    SPECS=("base:$INAT")
    for s in 0 1 2; do
        for cand in "$SSL_DIR/seed${s}_last.pth" \
                    "$SSL_DIR/leo_vitb16_ssl_seed${s}/checkpoints/last.pth" \
                    "$SSL_DIR/seed${s}/last.pth"; do
            [[ -f "$cand" ]] && { SPECS+=("ssl_seed${s}:$cand"); break; }
        done
    done
fi

echo "═══════════════════════════════════════════════"
echo "annot_extract | phases=$PHASES | threads=$THREADS"
echo "repo=$REPO | annot=$ANNOT | emb=$EMB"
echo "modèles :"
for spec in "${SPECS[@]}"; do
    echo "  - ${spec%%:*} → ${spec#*:}$( [[ -f "${spec#*:}" ]] || echo '   [MANQUANT]' )"
done
echo "═══════════════════════════════════════════════"

[[ -f "$REPO/annotations_leo/extract_embeddings.py" || -f "$ANNOT/extract_embeddings.py" ]] \
    || { echo "[ERREUR] extract_embeddings.py introuvable (ANNOT=$ANNOT)"; exit 1; }
[[ -f "$INAT" ]] || { echo "[ERREUR] checkpoint SimDINOv2-B absent : $INAT"; exit 1; }

for spec in "${SPECS[@]}"; do
    TAG="${spec%%:*}"; CKPT="${spec#*:}"
    [[ -f "$CKPT" ]] || { echo "[SKIP] $TAG : checkpoint absent ($CKPT)"; continue; }
    OUT="$EMB"
    [[ "$TAG" != "base" ]] && OUT="$EMB/$TAG"
    echo ""
    echo "─── $TAG ← $(basename "$CKPT") ───"
    ( cd "$ANNOT" && LEO_THREADS="$THREADS" "$PY" extract_embeddings.py \
        --phase "$PHASES" --ckpt "$CKPT" --out-dir "$OUT" )
    RC=$?
    [[ $RC -ne 0 ]] && echo "[WARN] $TAG a échoué (exit=$RC)" >&2
done

echo ""
echo "[annot_extract] terminé. Comparer ensuite :"
echo "  $PY $REPO/scripts/leo_ssl_annot_bench.py --clf logreg,knn,mlp --variants tile,fused"
echo "  (détecteurs pour l'outil d'annotation : --clf logreg --save-detectors)"
