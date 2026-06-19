#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Data curve Q5 — SLURM array Narval (6 proportions × 3 seeds séquentiels/tâche)
#
# AVANT SOUMISSION — éditer la section "Configuration" ci-dessous, puis :
#   sbatch scripts/slurm_datacurve.sh
#
# Structure : 6 tâches (une par proportion), chaque tâche exécute 3 seeds en série.
# Stratégie : 3 seeds séquentiels par A100 → 6 soumissions (vs 18), pas de collision
# mémoire inter-run, dataset partagé déjà dans $SLURM_TMPDIR.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=datacurve_vitb16
#SBATCH --array=0-5
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/datacurve_%A_%a.out
#SBATCH --error=logs/datacurve_%A_%a.err
# SBATCH --account=def-SUPERVISOR   # décommenter et éditer : votre compte Narval

# ─── Configuration — À ÉDITER ─────────────────────────────────────────────────
CODE_DIR="$HOME/Memoire"                          # Répertoire git clone sur Narval
TILES_SRC="/project/def-supervisor/$USER/tiles"  # Source des tuiles Arctic-TVC sur Narval
                                                   # (remplacer par votre chemin réel)
VENV="$CODE_DIR/.venv/bin/activate"               # Venv Python du projet
CONFIG="configs/vitb16_fulft_datacurve.yaml"
OUT_DIR="$SCRATCH/datacurve"                       # Scratch persistant (défini par Narval)

# Modèles HuggingFace pré-téléchargés (login node) — NE PAS télécharger sur nœud de calcul
export HF_HUB_OFFLINE=1                           # Force le mode offline
export HF_HOME="$HOME/.cache/huggingface"         # Cache HF (pré-rempli sur login node)
export TRANSFORMERS_OFFLINE=1

# Exposer CODE_DIR aux scripts Python (expansion des chemins Narval dans la config)
export CODE_DIR
# ──────────────────────────────────────────────────────────────────────────────

# Mapping index SLURM → fraction du train
FRACS=(0.01 0.05 0.10 0.50 0.70 1.00)
FRAC="${FRACS[$SLURM_ARRAY_TASK_ID]}"
PCT=$(python3 -c "print(int(round(${FRAC}*100)))")

echo "═══════════════════════════════════════════════"
echo "SLURM array task : $SLURM_ARRAY_TASK_ID → fraction=$FRAC (${PCT}%)"
echo "Nœud : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
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

# ── Environnement Python ──────────────────────────────────────────────────────
module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Copie des tuiles vers $SLURM_TMPDIR (fast local SSD) ─────────────────────
echo "[slurm] Copie des tuiles → $SLURM_TMPDIR/tiles ..."
mkdir -p "$SLURM_TMPDIR/tiles"
if [[ -d "$TILES_SRC" ]]; then
    rsync -a --info=progress2 "$TILES_SRC/" "$SLURM_TMPDIR/tiles/"
    echo "[slurm] Copie terminée."
else
    echo "[WARN] TILES_SRC non trouvé : $TILES_SRC — le dataset sera chargé depuis le réseau." >&2
    # Fallback : utiliser TILES_SRC directement
    export SLURM_TMPDIR_TILES_OVERRIDE="$TILES_SRC"
fi

# ── Archive RHOL (une seule fois si non existante) ────────────────────────────
# RHOL est exclu de l'entraînement Q5 ; tuiles archivées ici pour référence.
RHOL_ARCHIVE="$OUT_DIR/rhol_archive"
if [[ ! -d "$RHOL_ARCHIVE" ]]; then
    echo "[slurm] Archivage des tuiles RHOL → $RHOL_ARCHIVE ..."
    mkdir -p "$RHOL_ARCHIVE"
    find "$SLURM_TMPDIR/tiles" -path "*/RHOL/*" -name "*.png" \
        -exec cp --parents {} "$RHOL_ARCHIVE/" \;
    echo "[slurm] Archive RHOL : $(find "$RHOL_ARCHIVE" -name '*.png' | wc -l) tuiles"
fi

# ── Création des dossiers de sortie ──────────────────────────────────────────
mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# ── Boucle sur les 3 seeds (séquentielle, même A100) ─────────────────────────
for SEED in 0 1 2; do
    echo ""
    echo "─── Fraction=$FRAC  Seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] run frac=${FRAC} seed=${SEED} a échoué (exit=$EXIT_CODE)" >&2
        echo "  → Run loggé, continuation des autres seeds." >&2
        # Tier 11 : log, saute, continue (pas de retry infini)
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID (frac=$FRAC) terminée."
echo "  Résultats sous : $OUT_DIR/runs/"
echo "  Embeddings sous : $OUT_DIR/embeddings/"

# ── Vérification disk space (Tier 9) ─────────────────────────────────────────
EMB_SIZE=$(du -sh "$OUT_DIR/embeddings/" 2>/dev/null | cut -f1)
echo "[Tier9] Espace embeddings utilisé : $EMB_SIZE"
SCRATCH_FREE=$(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2{print $4}')
echo "[Tier9] Espace scratch disponible : $SCRATCH_FREE"
