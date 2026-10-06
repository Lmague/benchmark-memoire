#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# RAPATRIEMENT SSL Léo — par paliers, pour ne PAS perdre les artefacts coûteux.
#
# ⚠️ `$SCRATCH` n'existe pas dans le shell LOCAL. Chemin littéral sur Narval :
#    /scratch/lmague
#    Le scratch de l'Alliance est purgé après ~60 j d'inactivité : les checkpoints
#    SSL (12 Go, 81 GPU-h) sont l'artefact IRRÉMPLAÇABLE.
#
# Usage :
#   bash scripts/leo_ssl_harvest.sh 1        # résultats (≈20 Mo)  — à faire tout de suite
#   TIER=2 bash scripts/leo_ssl_harvest.sh   # + embeddings annotations (≈400 Mo)
#   TIER=3 bash scripts/leo_ssl_harvest.sh   # + checkpoints SSL (≈12 Go)
#   TIER=all bash scripts/leo_ssl_harvest.sh # tout sauf les 15 Go de sig_embeddings
#
# Le palier 3 est le plus important à long terme : 81 GPU-h non reproductibles.
# Le palier 4 (sig_embeddings Arctic, 15 Go) est VOLONTAIREMENT exclu : il se
# régénère en 1,3 h de GPU tant que les checkpoints existent.
# ═══════════════════════════════════════════════════════════════════════════════
set -u

TIER="${TIER:-${1:-1}}"
REMOTE_SCRATCH="${REMOTE_SCRATCH:-/scratch/lmague}"
REPO="${REPO:-$HOME/Documents/Mémoire}"
ANNOT="${ANNOT:-$HOME/annotations_leo}"
CKPT_LOCAL="${CKPT_LOCAL:-$REPO/checkpoints/leo_ssl}"
LOGS_LOCAL="$REPO/results/leo_explora_ssl/logs_narval"

say() { printf '\n\033[1m── %s\033[0m\n' "$*"; }
need() { [[ "$1" == "1" || "$TIER" == "2" || "$TIER" == "3" || "$TIER" == "all" ]]; }
has() { [[ "$TIER" == "$1" || "$TIER" == "all" ]]; }

mkdir -p "$REPO/results/leo_explora_ssl" "$LOGS_LOCAL"

if need 1; then
    say "1. Sonde d'alignement Arctic (JSON + tables, ~10 Mo)"
    rsync -avP "narval:$REMOTE_SCRATCH/leo_ssl/arctic_probe/probe_out/" \
        "$REPO/results/leo_explora_ssl/"

    say "1b. Trajectoire + banc des annotations (~100 Ko)"
    rsync -avP "narval:$REMOTE_SCRATCH/annot_leo/bench/" \
        "$REPO/results/leo_explora_ssl/annot_bench/"

    say "1c. meta.json de chaque tag Arctic (provenance, quelques ko)"
    rsync -avP --include='*/meta.json' --include='*/' --exclude='*.npy' \
        "narval:$REMOTE_SCRATCH/leo_ssl/arctic_probe/sig_embeddings/" \
        "$REPO/results/leo_explora_ssl/sig_meta/"

    say "1d. Logs des jobs (preuve d'exécution)"
    rsync -avP --include='leossl_*' --include='leo_ssl_4533845_*' --exclude='*' \
        "narval:$REMOTE_SCRATCH/../benchmark-memoire/logs/" "$LOGS_LOCAL/" 2>/dev/null \
      || ssh narval 'ls ~/benchmark-memoire/logs/leossl_*' 2>/dev/null | while read -r f; do
             rsync -avP "narval:$f" "$LOGS_LOCAL/"
         done
fi

if has 2; then
    say "2. Embeddings des annotations (~400 Mo : 34 tags × bank+negatives)"
    rsync -avP "narval:$REMOTE_SCRATCH/annot_leo/emb/" "$ANNOT/embeddings/"
    echo "   → utilisables tels quels par scripts/leo_ssl_annot_bench.py (--emb-dir $ANNOT/embeddings)"
fi

if has 3; then
    say "3. Checkpoints SSL Léo (12 Go — IRRÉMPLAÇABLE)"
    mkdir -p "$CKPT_LOCAL"
    for s in 0 1 2; do
        rsync -avP "narval:$REMOTE_SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed${s}/checkpoints/" \
            "$CKPT_LOCAL/seed${s}/"
    done
    echo "   → en local, tags attendus : seed0/ep004.pth … seed2/ep049.pth + last.pth"
    echo "   → repartir l'extraction Léo locale :"
    echo "     TAGS=\"base:\$REPO/checkpoints/simdinov2_vitb_inat21plantae.pth"
    echo "            ssl_seed0:\$CKPT_LOCAL/seed0/last.pth\" PHASES=bank bash scripts/leo_ssl_annot_extract.sh"
fi

say "MANIFESTE"
{
    echo "# Rapatriement SSL Léo — $(date -Iseconds)"
    echo "# palier TIER=$TIER | source narval:$REMOTE_SCRATCH"
    echo
    echo "## Fichiers rapatriés (taille, chemin relatif)"
    ( cd "$REPO/results/leo_explora_ssl" && find . -type f -printf '%10s  %p\n' | sort -k2 )
    [[ -d "$ANNOT/embeddings" ]] && ( cd "$ANNOT/embeddings" && find . -maxdepth 2 -type f -printf '%10s  %p\n' | sort -k2 | head -100 )
    [[ -d "$CKPT_LOCAL" ]] && ( cd "$CKPT_LOCAL" && find . -name '*.pth' -printf '%10s  %p\n' | sort -k2 )
} > "$REPO/results/leo_explora_ssl/MANIFEST_narval.txt" 2>/dev/null
cat "$REPO/results/leo_explora_ssl/MANIFEST_narval.txt" | head -40
echo
echo "  → manifeste complet : results/leo_explora_ssl/MANIFEST_narval.txt"
echo "  → espace disque : $(df -h "$REPO" | tail -1 | awk '{print $4}') libres sur $REPO"
