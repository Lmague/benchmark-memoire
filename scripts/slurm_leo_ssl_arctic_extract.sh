#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DOSE-RÉPONSE D'ALIGNEMENT — extraction frozen des checkpoints SSL Léo sur Arctic-TVC.
#
# Pour chaque seed (array 0-2) et chaque checkpoint sauvegardé par le SSL
# (`leo-explora-ssl/run_narval.sbatch`, toutes les 5 époques), on extrait les features
# frozen TUILE + CONTEXTE 512px sur les 3 splits Arctic-TVC → la sonde canonique
# locale produit ensuite les 3 compteurs d'alignement (tile / ctx / fused @512).
#
# Aucun entraînement supervisé : c'est un forward-only, donc léger en VRAM et
# repartable. Le point « époque 0 » (backbone SimDINOv2-B iNat d'origine) est extrait
# une seule fois (tag `leossl_b16_INAT_init`) : la LoRA du SSL est initialisée à zéro,
# donc teacher(step 0) ≡ SimB-iNat.
#
# PRÉ-REQUIS Narval :
#   $SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed{0,1,2}/checkpoints/ep*.pth
#   $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth   (point époque 0)
#   $SCRATCH/tiles.zip                                      (tuiles Arctic 224px)
#   $SCRATCH/context_512.zip  [+ context_512_valtest.zip]   (contexte 512px, cf.
#       scripts/slurm_merge_context_zip.sh — le zip « train-only » + le val/test)
#   ce dépôt à jour : $HOME/benchmark-memoire (git pull)
#
# Sorties : $SCRATCH/leo_ssl/arctic_probe/sig_embeddings/<tag>/{split}_{tile,ctx,labels}.npy
# Rapatriement (local) :
#   rsync -avP narval:$SCRATCH/leo_ssl/arctic_probe/sig_embeddings/ \
#       ~/Documents/Mémoire/results/leo_explora_ssl/sig_embeddings/
# puis : python scripts/leo_ssl_alignment_probe.py --workers 4
#
# Soumission : sbatch scripts/slurm_leo_ssl_arctic_extract.sh
#              (SEED=0 sbatch ... pour tester un seul seed)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=leossl_arctic
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/leossl_arctic_%A_%a.out
#SBATCH --error=logs/leossl_arctic_%A_%a.err
#SBATCH --account=def-bouguess_gpu

set -u
SEED="${SEED:-$SLURM_ARRAY_TASK_ID}"
CODE_DIR="$HOME/benchmark-memoire"
VENV="${VENV:-$HOME/ENV/bin/activate}"
TILES_ZIP="$SCRATCH/tiles.zip"
CTX_SIZE="${CTX_SIZE:-512}"
CTX_ZIP="$SCRATCH/context_${CTX_SIZE}.zip"
CTX_VT_ZIP="$SCRATCH/context_${CTX_SIZE}_valtest.zip"
CKPT_ROOT="$SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed${SEED}/checkpoints"
INAT_CKPT="$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"
OUT_DIR="$SCRATCH/leo_ssl/arctic_probe/sig_embeddings"
CSV_DIR=""
BATCH="${BATCH_OVERRIDE:-128}"
WORKERS="${WORKERS_OVERRIDE:-8}"
AMP_FLAG="${AMP_FLAG:-}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "leossl_arctic | seed=$SEED | ctx=${CTX_SIZE}px | batch=$BATCH"
echo "Job : ${SLURM_JOB_ID:-local} | Nœud : ${SLURMD_NODENAME:-?} | task : ${SLURM_ARRAY_TASK_ID:-?}"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ -f "$VENV" ]] || { echo "[ERROR] venv absent: $VENV (export VENV=...)"; exit 1; }
[[ -d "$CKPT_ROOT" ]] || { echo "[ERROR] $CKPT_ROOT absent — entraînement SSL pas rapatrié ?"; exit 1; }
[[ -f "$CTX_ZIP" ]] || { echo "[ERROR] $CTX_ZIP absent"; exit 1; }
[[ -f "$TILES_ZIP" ]] || { echo "[ERROR] $TILES_ZIP absent"; exit 1; }

# --- répertoire des CSV de split -------------------------------------------------- ⚠️ `spatial_datacurve/` a été RETIRÉ du dépôt (commit
# 0d0f6a4 : 95 fichiers, 1,4 M de lignes) — il n'existe plus que dans l'arbre de travail
# LOCAL. Sur Narval, seul `$SCRATCH/splits` est présent : c'est le split canonique v3,
# dont val.csv/test.csv sont md5-identiques à spatial_datacurve/splits/frac100_seed0/ et
# dont le train contient le MÊME ensemble de tuiles (ordre différent, sans effet ici :
# tous les tags lisent le même CSV, et la sonde ne dépend pas de l'ordre).
#
# Override manuel :  CSV_DIR_OVERRIDE=/chemin/vers/splits sbatch ...
CSV_CANDIDATES=(
    "${CSV_DIR_OVERRIDE:-}"
    "$CODE_DIR/spatial_datacurve/splits/frac100_seed0"
    "$CODE_DIR/splits"
    "$SCRATCH/splits"
)
for cand in "${CSV_CANDIDATES[@]}"; do
    [[ -n "$cand" && -f "$cand/test.csv" && -f "$cand/train.csv" ]] && { CSV_DIR="$cand"; break; }
done
if [[ -z "$CSV_DIR" ]]; then
    echo "[ERROR] aucun split trouvé. Candidats essayés :" >&2
    for cand in "${CSV_CANDIDATES[@]}"; do echo "        ${cand:-<vide>}" >&2; done
    echo "        → vérifier $SCRATCH/splits (attendu : train 49433 lignes)" >&2
    exit 1
fi
CANON_TEST_MD5="579abdf7b4a7bf64260d347fa1d998a5"   # splits/test.csv canonique v3
echo "[split] $CSV_DIR"
REAL_TEST_MD5=$(md5sum "$CSV_DIR/test.csv" | cut -d' ' -f1)
if [[ "$REAL_TEST_MD5" != "$CANON_TEST_MD5" ]]; then
    echo "[WARN] test.csv md5=$REAL_TEST_MD5 ≠ canonique $CANON_TEST_MD5" >&2
    echo "       → les F1 ne seront PAS directement comparables aux 0.5059 / 0.4931 /" >&2
    echo "         0.4717 du chapitre contexte (val/test doivent être ceux du split v3)." >&2
fi
# Le loader SimDINOv2 (src/models._ensure_sslplant_on_path) clone le fork vendor s'il est
# absent — or un nœud de calcul Narval n'a pas Internet. Échec rapide et explicite.
[[ -f "$CODE_DIR/vendors/sslplant/simdinov2/eval/get_model.py" ]] || {
    echo "[ERROR] vendors/sslplant absent dans $CODE_DIR."; 
    echo "        Sur le NŒUD DE LOGIN (Internet) : git clone --depth 1 https://github.com/ilyassmoummad/sslplant.git $CODE_DIR/vendors/sslplant";
    exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs "$OUT_DIR"

# --- données en RAM disque -----------------------------------------------------
echo "[slurm] Décompression tuiles + contexte ${CTX_SIZE}px → $SLURM_TMPDIR ..."
[[ -d "$SLURM_TMPDIR/tiles" ]] || unzip -q "$TILES_ZIP" -d "$SLURM_TMPDIR/"
CTX_DIR="$SLURM_TMPDIR/context_${CTX_SIZE}"
if [[ ! -d "$CTX_DIR" ]]; then
    unzip -q "$CTX_ZIP" -d "$SLURM_TMPDIR/"
    # val/test : zip complémentaire (tuiles disjointes, aucune collision de noms)
    [[ -f "$CTX_VT_ZIP" ]] && unzip -q "$CTX_VT_ZIP" -d "$SLURM_TMPDIR/"
fi
N_TILES=$(find "$SLURM_TMPDIR/tiles" -name '*.png' 2>/dev/null | wc -l)
N_CTX=$(find "$CTX_DIR" -name '*.png' 2>/dev/null | wc -l)
echo "[check] tuiles Arctic : $N_TILES (attendu ~80240) | contextes : $N_CTX"
[[ "$N_CTX" -lt 79000 ]] && echo "[WARN] contextes < 79k : la fusion context_${CTX_SIZE}.zip + _valtest.zip est peut-être incomplète" >&2

# --- point époque 0 (SimB iNat) : commun aux 3 seeds ----------------------------
if [[ -f "$INAT_CKPT" && ! -f "$OUT_DIR/leossl_b16_INAT_init/meta.json" ]]; then
    echo ""
    echo "─── baseline époque 0 : SimDINOv2-B iNat-Plantae ───"
    python scripts/leo_ssl_extract_arctic.py \
        --ckpt "$INAT_CKPT" --tag leossl_b16_INAT_init \
        --tiles-dir "$SLURM_TMPDIR/tiles" --context-dir "$CTX_DIR" \
        --csv-dir "$CSV_DIR" --out-dir "$OUT_DIR" --context-size "$CTX_SIZE" \
        --batch "$BATCH" --num-workers "$WORKERS" $AMP_FLAG \
        || echo "[WARN] extraction baseline échouée" >&2
fi

# --- checkpoints du seed --------------------------------------------------------
mapfile -t EPS < <(find "$CKPT_ROOT" -maxdepth 1 -name 'ep[0-9][0-9][0-9].pth' | sort)
if [[ ${#EPS[@]} -eq 0 ]]; then
    if [[ -f "$CKPT_ROOT/last.pth" ]]; then
        echo "[info] aucun ep*.pth — le run a été coupé avant la 1re sauvegarde (ep004) ; last.pth utilisé"
        EPS=("$CKPT_ROOT/last.pth")
    else
        echo "[ERROR] aucun checkpoint dans $CKPT_ROOT"; exit 1
    fi
fi
echo "[slurm] ${#EPS[@]} checkpoint(s) pour seed=$SEED : $(basename -a "${EPS[@]}" | tr '\n' ' ')"

for CK in "${EPS[@]}"; do
    BASE=$(basename "$CK" .pth)                    # ep004 | last
    TAG="leossl_b16_seed${SEED}_${BASE}"
    echo ""
    echo "─── $TAG ───"
    python scripts/leo_ssl_extract_arctic.py \
        --ckpt "$CK" --tag "$TAG" \
        --tiles-dir "$SLURM_TMPDIR/tiles" --context-dir "$CTX_DIR" \
        --csv-dir "$CSV_DIR" --out-dir "$OUT_DIR" --context-size "$CTX_SIZE" \
        --batch "$BATCH" --num-workers "$WORKERS" $AMP_FLAG
    RC=$?
    if [[ $RC -ne 0 ]]; then
        echo "[WARN] $TAG échoué (exit=$RC) — retry batch/2" >&2
        python scripts/leo_ssl_extract_arctic.py \
            --ckpt "$CK" --tag "$TAG" \
            --tiles-dir "$SLURM_TMPDIR/tiles" --context-dir "$CTX_DIR" \
            --csv-dir "$CSV_DIR" --out-dir "$OUT_DIR" --context-size "$CTX_SIZE" \
            --batch $((BATCH / 2)) --num-workers "$WORKERS" $AMP_FLAG \
            || echo "[WARN] retry $TAG échoué aussi — continuation" >&2
    fi
done

echo ""
echo "[slurm] seed=$SEED terminé → $OUT_DIR"
ls "$OUT_DIR" 2>/dev/null
echo "  Rapatriement : rsync -avP narval:$OUT_DIR/ results/leo_explora_ssl/sig_embeddings/"
echo "  Sonde locale : python scripts/leo_ssl_alignment_probe.py --workers 4"
echo "[slurm] done."
