#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI — schéma 8cls (8 classes) — CPU-only, Narval.
#
# Re-fitte chaque modèle (best_C lu dans 8cls/probe_knn.json, PAS de re-grille C),
# récupère (y_true, y_pred) sur le test, puis bootstrappe le test set (1000 tirages)
# → IC95 par percentiles + comparaison appariée gelé vs fine-tuné (f1_macro_pres).
#
# IMPORTANT (--pass 8cls) : le re-fit applique la MÊME réduction de classes que
# probe.py (drop_class cascade source-aware). Sur 8cls, f1_macro_all == f1_macro_pres
# (les 8 classes sont toutes présentes dans le test). Voir known_issues.md.
#
# AUCUN GPU (sklearn CPU sur embeddings déjà extraits).
# Soumission :  git pull && sbatch scripts/slurm_bootstrap_8cls.sh
#
# Pré-requis présents sous $SCRATCH :
#   $SCRATCH/embeddings/                         → embeddings canoniques
#   $SCRATCH/sota_screening/                     → runs SOTA (si comparés via --pairs)
#   $SCRATCH/datacurve/results/8cls/probe_knn.json → best_C par modèle (96 modèles)
# Sortie :
#   $SCRATCH/datacurve/results/8cls/bootstrap_ci.json
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=bootstrap_8cls
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=logs/bootstrap_8cls_%j.out
#SBATCH --error=logs/bootstrap_8cls_%j.err
#SBATCH --account=def-bouguess   # ← éditer : votre compte Alliance/Narval
# PAS de --gres=gpu : 100 % CPU. Le run schéma A (12 modèles canon.+FT) prend ~30 min ;
# 1 h de marge. Si l'on ajoute des runs SOTA via --pairs, augmenter --time en conséquence.

# ─── Configuration — À ÉDITER ─────────────────────────────────────────────────
CONFIG="configs/probe_all.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
PROBE_JSON="$SCRATCH/datacurve/results/8cls/probe_knn.json"
OUTPUT="$SCRATCH/datacurve/results/8cls/bootstrap_ci.json"
N_BOOTSTRAP=1000

# Sanity check reproductibilité AVANT le corpus : l'observed du bootstrap doit matcher
# le probe 8cls (dinov3_vitl16_lvd test.f1_macro_all = 0.6650, cf. 8cls/probe_knn.json ;
# sur 8cls f1_macro_all == f1_macro_pres car les 8 classes sont présentes).
SANITY_MODEL="dinov3_vitl16_lvd"
SANITY_TOL=0.0005   # ~1e-4 (même best_C, même fit → reproduction quasi bit-pour-bit)

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Threads BLAS/sklearn = cœurs alloués (le fit lbfgs est BLAS-bound).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════"
echo "Bootstrap CI 8cls (CPU)  | Job : ${SLURM_JOB_ID:-local}"
echo "Nœud : ${SLURMD_NODENAME:-?}  | cœurs : ${SLURM_CPUS_PER_TASK:-?}"
echo "═══════════════════════════════════════════════"

if [[ ! -d "$CODE_DIR" ]]; then echo "[ERROR] CODE_DIR absent : $CODE_DIR" >&2; exit 1; fi
if [[ ! -f "$VENV" ]];    then echo "[ERROR] venv absent : $VENV" >&2;      exit 1; fi

module load python/3.11
source "$VENV"
cd "$CODE_DIR"
mkdir -p "$CODE_DIR/logs"

if [[ ! -f "$PROBE_JSON" ]]; then
    echo "[ERROR] probe 8cls introuvable : $PROBE_JSON (lancer slurm_probe_all.sh d'abord)." >&2
    exit 1
fi

# ── ÉTAPE 1 — SANITY CHECK reproductibilité (STOP si échec) ──────────────────
echo ""
echo "─── Sanity : $SANITY_MODEL / 8cls — bootstrap observed vs probe_knn.json ───"
python scripts/bootstrap_ci.py --config "$CONFIG" --pass 8cls \
    --probe-json "$PROBE_JSON" --n-bootstrap 10 \
    --output "/tmp/bootstrap_8cls_sanity.json"
SANITY_EXIT=$?
if [[ $SANITY_EXIT -ne 0 ]]; then
    echo "[ERROR] bootstrap de sanity a échoué (exit=$SANITY_EXIT) — corpus NON lancé." >&2
    exit 1
fi

python3 - "$SANITY_MODEL" "$PROBE_JSON" "$SANITY_TOL" <<'PY'
import json, sys
model, probe_json, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
boot = json.load(open("/tmp/bootstrap_8cls_sanity.json"))
probe = json.load(open(probe_json))
probe = probe.get("probe", probe)
if model not in boot["models"]:
    print(f"[SANITY][FAIL] {model} absent du bootstrap (skippé ?).", file=sys.stderr); sys.exit(2)
obs = boot["models"][model]["f1_macro_pres"]["observed"]
ref = probe[model]["test"]["f1_macro_pres"]   # == f1_macro_all en 8cls
delta = abs(obs - ref)
print(f"[SANITY] {model} 8cls  bootstrap observed={obs:.4f}  probe={ref:.4f}  |Δ|={delta:.5f} (tol={tol})")
if delta > tol:
    print(f"[SANITY][FAIL] écart {delta:.5f} > tol {tol} — le re-fit a DÉVIÉ du probe. "
          f"Corpus NON lancé.", file=sys.stderr); sys.exit(3)
print("[SANITY][OK] reproduction confirmée → lancement du corpus complet.")
PY
if [[ $? -ne 0 ]]; then echo "[ERROR] sanity NON conforme — corpus NON lancé." >&2; exit 1; fi

# ── ÉTAPE 2 — CORPUS COMPLET (12 canoniques + fine-tunés, 1000 tirages) ──────
echo ""
echo "─── Corpus 8cls : canoniques + fine-tunés × ${N_BOOTSTRAP} tirages ───"
python scripts/bootstrap_ci.py --config "$CONFIG" --pass 8cls \
    --probe-json "$PROBE_JSON" --include-finetuned \
    --n-bootstrap "$N_BOOTSTRAP" --output "$OUTPUT"
FULL_EXIT=$?

echo ""
echo "═══════════════════════════════════════════════"
if [[ $FULL_EXIT -eq 0 ]]; then
    echo "[bootstrap_8cls] TERMINÉ → $OUTPUT"
else
    echo "[bootstrap_8cls] ÉCHEC (exit=$FULL_EXIT)."
fi
echo "═══════════════════════════════════════════════"

# ── (Optionnel) inclure des runs SOTA dans la comparaison via --pairs ────────
# Les clés SOTA (vitb16_{regime}_frac{XXX}_seed{N}) sont routées vers load_sota_features
# (is_sota_key) et réduites 11cls→8cls (drops=[7,3,1]). Exemple :
#   python scripts/bootstrap_ci.py --config "$CONFIG" --pass 8cls \
#       --probe-json "$PROBE_JSON" --include-finetuned --n-bootstrap "$N_BOOTSTRAP" \
#       --pairs dinov3_vitl16_lvd:vitb16_full_frac100_seed0 \
#       --output "$OUTPUT" --pairs-output "$SCRATCH/datacurve/results/8cls/bootstrap_pairs.json"

exit $FULL_EXIT
