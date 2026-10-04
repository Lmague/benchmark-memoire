#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION SSL Léo SUR LES ANNOTATIONS — job unique, à lancer APRÈS les 3 seeds SSL.
#
# Enchaîne sur Narval, sans raster (les 4 photos sont des JPEG 224 déjà dans
# $SCRATCH/web/tiles/clairiere) :
#   1. embeddings des points annotés + des négatifs, pour la baseline SimDINOv2-B
#      ET les 3 checkpoints ExPLoRA-Léo  → $SCRATCH/annot_leo/emb/<tag>/{bank,negatives}.npz
#   2. banc d'évaluation apparié (logreg / kNN) : multiclass + détection par espèce,
#      somme des probas des 3 seeds (ensemble) vs chaque seed vs mean±std
#      → $SCRATCH/annot_leo/bench/{bench_annot.json,bench_annot.md}
#
# PRÉ-REQUIS (envoyés depuis le local — voir README §3) :
#   $SCRATCH/annot_leo/annot_index.npz            (scripts/leo_ssl_annot_prepare_index.py)
#   $SCRATCH/annot_leo/session.gpkg               (points annotés)
#   $SCRATCH/annot_leo/candidates.sqlite          (candidats rejetés = négatifs durs)
#   $SCRATCH/annot_leo/manifest_clairiere.json    (idx -> chemin de tuile)
#   $SCRATCH/web/tiles/clairiere/**/*.jpg         (déjà en place)
#   $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth          (baseline)
#   $SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed{0,1,2}/checkpoints/last.pth
#   $HOME/benchmark-memoire/vendors/sslplant/     (clone sur le nœud de LOGIN)
#
# LANCEMENT (après soumission de l'entraînement SSL) :
#   SSLID=$(squeue -u $USER -n leo_explora_ssl -h -o %A | head -1)
#   sbatch --dependency=afterok:$SSLID scripts/slurm_leo_ssl_annot_eval.sh
#
# Rapatriement :
#   rsync -avP narval:$SCRATCH/annot_leo/bench/ results/leo_explora_ssl/annot_bench/
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=leossl_annot
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/leossl_annot_%j.out
#SBATCH --error=logs/leossl_annot_%j.err
#SBATCH --account=def-bouguess_gpu

set -u
CODE_DIR="$HOME/benchmark-memoire"
VENV="${VENV:-$HOME/ENV/bin/activate}"
WEB="$SCRATCH/web"
ANNOT="$SCRATCH/annot_leo"
EMB="$ANNOT/emb"
BENCH="$ANNOT/bench"
BATCH="${BATCH_OVERRIDE:-256}"
WORKERS="${WORKERS_OVERRIDE:-8}"
CLF="${CLF:-logreg,knn}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "leossl_annot | job $SLURM_JOB_ID | nœud ${SLURMD_NODENAME:-?}"
echo "═══════════════════════════════════════════════"

[[ -d "$CODE_DIR" ]] || { echo "[ERROR] CODE_DIR absent : $CODE_DIR"; exit 1; }
[[ -f "$VENV" ]] || { echo "[ERROR] venv absent : $VENV (export VENV=...)"; exit 1; }
[[ -f "$CODE_DIR/vendors/sslplant/simdinov2/eval/get_model.py" ]] || {
    echo "[ERROR] vendors/sslplant absent — sur le NŒUD DE LOGIN :"
    echo "        git clone --depth 1 https://github.com/ilyassmoummad/sslplant.git $CODE_DIR/vendors/sslplant"
    exit 1; }
[[ -d "$WEB/tiles/clairiere" ]] || { echo "[ERROR] $WEB/tiles/clairiere absent"; exit 1; }
for f in annot_index.npz manifest_clairiere.json; do
    [[ -f "$ANNOT/$f" ]] || { echo "[ERROR] $ANNOT/$f absent (rsync depuis le local — README §3)"; exit 1; }
done
# session.gpkg / candidates.sqlite : utiles seulement pour la provenance (l'index les a déjà résolus)
for f in session.gpkg candidates.sqlite; do
    [[ -f "$ANNOT/$f" ]] || echo "[info] $ANNOT/$f absent (facultatif : l'index contient déjà les points)"
done
[[ -f "$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth" ]] || {
    echo "[ERROR] checkpoint baseline SimDINOv2-B absent"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs "$EMB" "$BENCH"

declare -A CKPTS
CKPTS[base]="$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"
# ALL_EPOCHS=1 : tous les ep*.pth (trajectoire) au lieu du seul last.pth.
# Tags : ssl_seed{S} (défaut) ou ssl_seed{S}_{ep} (ALL_EPOCHS).
ALL_EPOCHS="${ALL_EPOCHS:-0}"
TAGS_ORDER=("base")
for s in 0 1 2; do
    if [[ "$ALL_EPOCHS" == "1" ]]; then
        for ep in "$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed${s}/checkpoints"/ep[0-9][0-9][0-9].pth; do
            [[ -f "$ep" ]] || continue
            b=$(basename "$ep" .pth)
            CKPTS["ssl_seed${s}_${b}"]="$ep"
            TAGS_ORDER+=("ssl_seed${s}_${b}")
        done
    else
        CKPTS["ssl_seed${s}"]="$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed${s}/checkpoints/last.pth"
        TAGS_ORDER+=("ssl_seed${s}")
    fi
done
echo "[config] ALL_EPOCHS=$ALL_EPOCHS → ${#TAGS_ORDER[@]} tags"

# ── 1. extraction (GPU) ─────────────────────────────────────────────────────────
FAILED=()
for tag in "${TAGS_ORDER[@]}"; do
    CK="${CKPTS[$tag]}"
    if [[ ! -f "$CK" ]]; then
        echo "[SKIP] $tag : checkpoint absent ($CK)"
        FAILED+=("$tag")
        continue
    fi
    echo ""
    echo "─── extraction $tag ← $(basename "$(dirname "$(dirname "$CK")")")/$(basename "$CK") ───"
    python scripts/leo_ssl_annot_extract_web.py \
        --ckpt "$CK" --tag "$tag" \
        --index "$ANNOT/annot_index.npz" \
        --web-dir "$WEB" --site clairiere \
        --manifest "$ANNOT/manifest_clairiere.json" \
        --out-dir "$EMB" --batch "$BATCH" --num-workers "$WORKERS" --amp \
        || { echo "[WARN] extraction $tag échouée" >&2; FAILED+=("$tag"); }
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "[WARN] tags en échec : ${FAILED[*]} — vérifier les checkpoints SSL rapatriés"
fi
N_OK=$(ls -d "$EMB"/*/ 2>/dev/null | wc -l)
echo "[check] $N_OK tag(s) extraits dans $EMB"

# ── 2. banc d'évaluation (CPU, mono-thread BLAS) ────────────────────────────────
echo ""
if [[ "$ALL_EPOCHS" == "1" ]]; then
    echo "─── trajectoire (tous les checkpoints) ───"
    python scripts/leo_ssl_annot_trajectory.py \
        --emb-dir "$EMB" --out-dir "$BENCH" --clf "$CLF" --task both \
        || { echo "[ERROR] trajectoire échouée" >&2; exit 1; }
    REPORT="$BENCH/trajectory.md"
else
    echo "─── banc d'évaluation (clf=$CLF, variante tile) ───"
    python scripts/leo_ssl_annot_bench.py \
        --emb-dir "$EMB" --out-dir "$BENCH" \
        --clf "$CLF" --variants tile --task both \
        || { echo "[ERROR] bench échoué" >&2; exit 1; }
    REPORT="$BENCH/bench_annot.md"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "[slurm] terminé."
ls -la "$BENCH" 2>/dev/null
echo "  Rapport : $REPORT"
echo "  Rapatriement : rsync -avP narval:$BENCH/ ~/Documents/Mémoire/results/leo_explora_ssl/annot_bench/"
echo "[slurm] done."
