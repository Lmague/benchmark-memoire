#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SONDE D'ALIGNEMENT — job CPU sur Narval (les embeddings sont DÉJÀ sur $SCRATCH).
#
# Pourquoi sur Narval : 93 sondes (31 tags × 3 variantes) à ~22 min de CPU mono-cœur
# chacune = 11,4 h de calcul (23 h avec --ensemble). La parallélisation se fait par
# PROCESSUS (BLAS reste mono-thread, cf. AGENTS.md §4.8), donc un nœud à 32 cœurs fait
# le travail en ~45 min. Et surtout : les 15 Go de sig_embeddings sont déjà là — on ne
# rapatrie que les ~10 Mo de JSON.
#
# AUCUN GPU : c'est du sklearn (LogisticRegression lbfgs), CPU pur.
#
# PRÉ-REQUIS : le job scripts/slurm_leo_ssl_arctic_extract.sh doit avoir tourné
#              (31 tags dans $SCRATCH/leo_ssl/arctic_probe/sig_embeddings/).
#
# Soumission : sbatch scripts/slurm_leo_ssl_alignment_probe.sh
#   ALL=0 pour un premier passage rapide (sans l'ensemble des seeds, ~23 min) :
#       ENSEMBLE=0 sbatch scripts/slurm_leo_ssl_alignment_probe.sh
#
# Rapatriement (petit !) :
#   rsync -avP narval:$SCRATCH/leo_ssl/arctic_probe/probe_out/ results/leo_explora_ssl/
# puis en local :  python scripts/leo_ssl_alignment_plot.py
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=leossl_probe
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/leossl_probe_%j.out
#SBATCH --error=logs/leossl_probe_%j.err
#SBATCH --account=def-bouguess
# ⚠️ Compte CPU, PAS `def-bouguess_gpu` (ce job ne demande aucun GPU).
#    Vérifier avec :  sacctmgr show assoc user=$USER format=account%30,partition

set -u
CODE_DIR="$HOME/benchmark-memoire"
VENV="${VENV:-$HOME/ENV/bin/activate}"
SIG_DIR="$SCRATCH/leo_ssl/arctic_probe/sig_embeddings"
OUT_DIR="$SCRATCH/leo_ssl/arctic_probe/probe_out"
WORKERS="${WORKERS_OVERRIDE:-$SLURM_CPUS_PER_TASK}"
ENSEMBLE="${ENSEMBLE:-1}"

echo "═══════════════════════════════════════════════"
echo "leossl_probe | job $SLURM_JOB_ID | nœud ${SLURMD_NODENAME:-?}"
echo "workers=$WORKERS | ensemble=$ENSEMBLE"
echo "═══════════════════════════════════════════════"

[[ -d "$CODE_DIR" ]] || { echo "[ERROR] CODE_DIR absent : $CODE_DIR"; exit 1; }
[[ -f "$VENV" ]] || { echo "[ERROR] venv absent : $VENV"; exit 1; }
[[ -d "$SIG_DIR" ]] || { echo "[ERROR] $SIG_DIR absent — lancer d'abord le job d'extraction"; exit 1; }

module load python/3.11
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs "$OUT_DIR"

N_TAGS=$(ls -d "$SIG_DIR"/*/ 2>/dev/null | wc -l)
echo "[check] $N_TAGS tag(s) dans $SIG_DIR (attendu 31 : 1 baseline + 3 seeds × 10)"
[[ "$N_TAGS" -lt 31 ]] && echo "[WARN] moins de 31 tags — la dose-réponse sera incomplète" >&2

# Éviter que worker × BLAS threads = explosion : le script force déjà le mono-thread,
# on insiste ici pour les sous-processus.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

ARGS=(--sig-dir "$SIG_DIR" --out-dir "$OUT_DIR" --workers "$WORKERS")
[[ "$ENSEMBLE" == "1" ]] && ARGS+=(--ensemble)

echo "[run] python scripts/leo_ssl_alignment_probe.py ${ARGS[*]}"
python scripts/leo_ssl_alignment_probe.py "${ARGS[@]}"
RC=$?
[[ $RC -ne 0 ]] && { echo "[ERROR] sonde échouée (exit=$RC)" >&2; exit $RC; }

echo ""
echo "═════════════════ RÉSULTAT ═════════════════"
cat "$OUT_DIR/alignment_summary.md" 2>/dev/null || echo "(summary absent)"
echo "═══════════════════════════════════════════"
echo ""
echo "[slurm] Rapatriement (petit, ~10 Mo) :"
echo "  rsync -avP narval:$OUT_DIR/ results/leo_explora_ssl/"
echo "[slurm] done."
