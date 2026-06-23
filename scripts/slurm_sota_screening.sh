#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# SoTA screening — baseline FT supervisée Arctic-TVC (11 classes, RHOL exclu).
# 4 régimes × 3 seeds = 12 runs à 100% des données (phase screening avant scaling).
#
#   array 0 → full    (vitb16_fulft_sota.yaml)   — full FT amélioré (recette T1)
#   array 1 → mhsa    (vitb16_mhsa_sota.yaml)    — attention + head seulement
#   array 2 → explora (vitb16_explora_sota.yaml) — ExPLoRA-like supervisé (LoRA Q,V + 2 blocs full)
#   array 3 → scratch (vitb16_scratch_sota.yaml) — init aléatoire (BORNE INFÉRIEURE)
#
# Chaque tâche array = 1 régime, 3 seeds séquentiels sur la même A100.
# Soumission :  sbatch scripts/slurm_sota_screening.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=sota_screening
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/sota_%A_%a.out
#SBATCH --error=logs/sota_%A_%a.err
#SBATCH --account=def-bouguess   # ← éditer : votre compte Alliance/Narval
# NOTE : scratch (array 3 : 100 epochs × 3 seeds) est le plus long. S'il atteint la limite
#        de 10h, resoumettre uniquement cette tâche avec un --time plus grand :
#        sbatch --array=3 --time=20:00:00 scripts/slurm_sota_screening.sh
#        (--skip-if-done saute les seeds déjà complétés ; un seed interrompu repart à zéro.)

# ─── Configuration — À ÉDITER ─────────────────────────────────────────────────
CONFIGS=(
  "configs/vitb16_fulft_sota.yaml"
  "configs/vitb16_mhsa_sota.yaml"
  "configs/vitb16_explora_sota.yaml"
  "configs/vitb16_scratch_sota.yaml"
)
REGIME_TAGS=(full mhsa explora scratch)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
REGIME_TAG="${REGIME_TAGS[$SLURM_ARRAY_TASK_ID]}"

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
# OUT_DIR par RÉGIME → aucune collision de run-dir entre régimes (les ckpt_tag/emb_key
# encodent déjà le régime, double sécurité — cf. datacurve_one_run._REGIME_TAG).
OUT_DIR="$SCRATCH/sota_screening/${REGIME_TAG}"
FRAC=1.00   # screening = 100% des données

# Modèles HuggingFace/timm pré-téléchargés (login node) — pas de download sur nœud calcul
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════"
echo "SLURM array task $SLURM_ARRAY_TASK_ID → régime=$REGIME_TAG  config=$CONFIG"
echo "Nœud : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

# ── Vérifications préliminaires ───────────────────────────────────────────────
if [[ -z "$CONFIG" ]]; then
    echo "[ERROR] index array $SLURM_ARRAY_TASK_ID hors plage (attendu 0-3)" >&2
    exit 1
fi
if [[ ! -d "$CODE_DIR" ]]; then
    echo "[ERROR] CODE_DIR non trouvé : $CODE_DIR" >&2
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] venv non trouvé : $VENV" >&2
    exit 1
fi

# ── Environnement Python ──────────────────────────────────────────────────────
module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Copie/extraction des tuiles → $SLURM_TMPDIR (SSD local rapide) ────────────
echo "[slurm] Extraction des tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] Extraction terminée : $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable" >&2
    exit 1
fi

# ── Création des dossiers de sortie ──────────────────────────────────────────
mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# ── Boucle sur les 3 seeds (séquentielle, même A100) ─────────────────────────
for SEED in 0 1 2; do
    echo ""
    echo "─── Régime=$REGIME_TAG  Fraction=$FRAC  Seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] run régime=$REGIME_TAG seed=$SEED a échoué (exit=$EXIT_CODE)" >&2
        echo "  → loggé, continuation des autres seeds." >&2
    fi
done

# ── Résumé f1_macro_pres_test des 3 seeds ────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "[résumé] Régime=$REGIME_TAG — f1_macro_pres_test par seed"
echo "═══════════════════════════════════════════════"
PCT=$(python3 -c "print(int(round(${FRAC}*100)))")
python3 - "$OUT_DIR" "$PCT" "$REGIME_TAG" <<'PY'
import json, os, sys
out_dir, pct, regime = sys.argv[1], int(sys.argv[2]), sys.argv[3]
vals = []
for seed in (0, 1, 2):
    mp = os.path.join(out_dir, "runs", f"frac{pct:03d}_seed{seed}", "metrics.json")
    if not os.path.exists(mp):
        print(f"  seed{seed}: (metrics.json absent — run incomplet)")
        continue
    m = json.load(open(mp))
    f1 = m.get("f1_macro_pres_test")
    f8 = m.get("f1_macro_8cls_test")
    vals.append(f1)
    print(f"  seed{seed}: f1_macro_pres_test={f1:.4f}  f1_macro_8cls_test={f8:.4f}  best_C={m.get('best_C')}")
if vals:
    import statistics as st
    mean = sum(vals) / len(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  → [{regime}] f1_macro_pres_test = {mean:.4f} ± {sd:.4f}  (n={len(vals)} seeds)")
else:
    print(f"  → [{regime}] aucun run complété.")
PY

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID ($REGIME_TAG) terminée."
echo "  Résultats : $OUT_DIR/runs/   |   Embeddings : $OUT_DIR/embeddings/"
SCRATCH_FREE=$(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2{print $4}')
echo "[slurm] Espace scratch disponible : $SCRATCH_FREE"
