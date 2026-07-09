#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Probe linéaire + k-NN CANONIQUE sur le CORPUS ÉLARGI (CPU-only, Narval).
#
#   - 12 modèles canoniques (9 frozen + 3 fine-tunés)        → schéma 12cls
#   - 84 runs SOTA (4 régimes × 7 fractions × 3 seeds)       → schéma 11cls_no_rhol
#   × 3 passes de classes : with_rhol (12cls), without_rhol (11cls), 8cls (8cls).
#
# AUCUN GPU (probe = sklearn CPU sur embeddings déjà extraits).
# Soumission :  git pull && sbatch scripts/slurm_probe_all.sh
#
# Pré-requis présents sous $SCRATCH (aucune extraction ici) :
#   $SCRATCH/embeddings/         → embeddings canoniques ({model}_{split}.npy)
#   $SCRATCH/sota_screening/     → runs SOTA ({regime}/embeddings/{run}/{split}.npy)
# Sortie :
#   $SCRATCH/datacurve/results/{with_rhol,without_rhol,8cls}/probe_knn.json
#   (with_rhol ne contient QUE les 12 canoniques : RHOL absent des runs SOTA.)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=probe_all
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=logs/probe_all_%j.out
#SBATCH --error=logs/probe_all_%j.err
#SBATCH --account=def-bouguess   # ← éditer : votre compte Alliance/Narval
# PAS de --gres=gpu : ce job est 100 % CPU (linear probe sklearn).
# NOTE durée : ~96 modèles × 3 passes. Si 8 h insuffisant, relancer le même sbatch :
#   les passes déjà écrites (probe_knn.json présent) sont SAUTÉES (idempotence), donc
#   la reprise repart de la première passe non terminée. Pour du grain plus fin,
#   scinder par régime avec --only "$(...)" + --merge (voir plus bas).

# ─── Configuration — À ÉDITER ─────────────────────────────────────────────────
CONFIG="configs/probe_all.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"

# Sanity check reproductibilité (garde-fou AVANT le corpus complet).
# dinov3_vitb16_lvd / without_rhol / f1_macro_all = 0.4712 (valeur canonique, reproduite
# bit-pour-bit en local ; cf. results/without_rhol/probe_knn_cgrid.json et relance2_11cls.json).
# NB : la valeur 0.4791 mentionnée ailleurs correspond à dinov3_vitl16_lvd (0.4792), pas au ViT-B.
SANITY_MODEL="dinov3_vitb16_lvd"
SANITY_EXPECTED=0.4712
SANITY_TOL=0.010

# Modèles HuggingFace/timm : non requis (probe lit uniquement des .npy), mais on force
# l'offline par cohérence avec les autres jobs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CODE_DIR
# Threads BLAS/sklearn = cœurs alloués (le fit lbfgs est BLAS-bound).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════"
echo "Probe corpus élargi (CPU)  | Job : ${SLURM_JOB_ID:-local}"
echo "Nœud : ${SLURMD_NODENAME:-?}  | cœurs : ${SLURM_CPUS_PER_TASK:-?}"
echo "═══════════════════════════════════════════════"

# ── Vérifications préliminaires ───────────────────────────────────────────────
if [[ ! -d "$CODE_DIR" ]]; then
    echo "[ERROR] CODE_DIR non trouvé : $CODE_DIR" >&2
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] venv non trouvé : $VENV" >&2
    exit 1
fi

module load python/3.11
source "$VENV"
cd "$CODE_DIR"
mkdir -p "$CODE_DIR/logs"

if [[ ! -d "$SCRATCH/embeddings" ]]; then
    echo "[ERROR] $SCRATCH/embeddings introuvable (embeddings canoniques requis)." >&2
    exit 1
fi
if [[ ! -d "$SCRATCH/sota_screening" ]]; then
    echo "[ERROR] $SCRATCH/sota_screening introuvable (runs SOTA requis)." >&2
    exit 1
fi

# ── ÉTAPE 1 — SANITY CHECK reproductibilité (STOP si échec) ──────────────────
echo ""
echo "─── Sanity check : $SANITY_MODEL / without_rhol (attendu ${SANITY_EXPECTED} ± ${SANITY_TOL}) ───"
python probe.py --config "$CONFIG" --only "$SANITY_MODEL" --output-tag sanity --force
SANITY_EXIT=$?
if [[ $SANITY_EXIT -ne 0 ]]; then
    echo "[ERROR] le probe de sanity a échoué (exit=$SANITY_EXIT) — corpus complet NON lancé." >&2
    exit 1
fi

python3 - "$SANITY_MODEL" "$SANITY_EXPECTED" "$SANITY_TOL" <<'PY'
import json, os, sys
model, expected, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
results_dir = os.environ.get("SCRATCH", ".") + "/datacurve/results"
path = os.path.join(results_dir, "without_rhol", "probe_knn_sanity.json")
if not os.path.exists(path):
    print(f"[SANITY][FAIL] {path} absent — impossible de vérifier.", file=sys.stderr)
    sys.exit(2)
d = json.load(open(path))
f1 = d["probe"][model]["test"]["f1_macro_all"]
delta = abs(f1 - expected)
print(f"[SANITY] {model} without_rhol f1_macro_all = {f1:.4f} "
      f"(attendu {expected:.4f}, |Δ|={delta:.4f}, tol={tol:.4f})")
if delta > tol:
    print(f"[SANITY][FAIL] écart {delta:.4f} > tol {tol:.4f} — le pipeline a DÉVIÉ de la "
          f"référence canonique. Corpus complet NON lancé.", file=sys.stderr)
    sys.exit(3)
print("[SANITY][OK] reproduction canonique confirmée → lancement du corpus complet.")
PY
SANITY_CHECK=$?
if [[ $SANITY_CHECK -ne 0 ]]; then
    echo "[ERROR] sanity check NON conforme (code=$SANITY_CHECK) — corpus complet NON lancé." >&2
    exit 1
fi

# ── ÉTAPE 2 — CORPUS COMPLET (12 canoniques + 84 SOTA, 3 passes) ─────────────
echo ""
echo "─── Corpus complet : 12 canoniques + 84 SOTA × 3 passes ───"
python probe.py --config "$CONFIG" --include-sota --force
FULL_EXIT=$?

echo ""
echo "═══════════════════════════════════════════════"
if [[ $FULL_EXIT -eq 0 ]]; then
    echo "[probe_all] TERMINÉ. Résultats :"
    for TAG in with_rhol without_rhol 8cls; do
        echo "  $SCRATCH/datacurve/results/$TAG/probe_knn.json"
    done
else
    echo "[probe_all] ÉCHEC ou interruption (exit=$FULL_EXIT). Relancer le même sbatch :"
    echo "  les passes déjà écrites seront sautées (idempotence)."
fi
echo "═══════════════════════════════════════════════"

# ── (Optionnel) reprise grain fin par régime, en cas de time-out mi-passe ────
# Chaque appel MERGE dans les JSON de passe existants (progrès persistant après
# chaque régime). Décommenter et lancer manuellement au besoin :
#
#   # d'abord les canoniques seuls :
#   python probe.py --config configs/probe_all.yaml --force
#   # puis un régime SOTA à la fois (21 runs) :
#   for R in full mhsa explora scratch; do
#     KEYS=$(python3 -c "print(' '.join(f'vitb16_${R}_frac{f}_seed{s}' \
#       for f in ['001','005','010','025','050','070','100'] for s in [0,1,2]))")
#     python probe.py --config configs/probe_all.yaml --only $KEYS --merge
#   done

exit $FULL_EXIT
