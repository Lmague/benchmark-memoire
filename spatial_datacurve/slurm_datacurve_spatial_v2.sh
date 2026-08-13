#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Data curve SPATIALE v2 — DINOv3-B LoRA r=8 alpha=16 (Q,V sur tous les blocs)
#
# Courbe de quantité de données contrôlée par unités SPATIALES (orthomosaïques
# entières ≥ 25 % / blocs spatiaux contigus < 25 %), PAS par tuiles aléatoires.
# Le volume de train est déjà encodé dans les CSV de
# `spatial_datacurve/splits/fracXXX_seed{S}/` (construits par
# `spatial_datacurve/make_spatial_datacurve.py`, algo closest-sum pour les
# orthos entières) : on entraîne donc sur la TOTALITÉ du CSV spatial en passant
# `--fraction 1.0` (aucun sous-échantillonnage stratifié interne).
#
# 7 fractions × 3 seeds = 21 runs. Array de 7 jobs (une fraction chacun),
# 3 seeds SÉQUENTIELS par job (même GPU, même extraction de tuiles).
#
# GPU : `a100_2g.10gb:1` = DEMI-A100 (10 Go) — on ne prend pas de GPU entier,
# même convention que slurm_datacurve_lora.sh / slurm_datacurve_spatial.sh (v1).
#
# Config : configs/dinov3_lora_datacurve_r8a16.yaml (r=8, alpha=16 = 2r) — tag
# de run `dinov3_vitb16_lvd_lora_r8a16_frac100_seed{N}` = identique aux runs
# datacurve aléatoire existants (comparabilité).
#
# NB — artefact de tag : le pipeline dérive le tag de `--fraction` → tous les
# niveaux spatiaux portent le suffixe `frac100_seed{N}`. On évite toute
# collision en scopant `--out-dir` et `--emb-dir` par fraction. Le volume RÉEL
# est enregistré dans `metrics.json` (clé `n_train_tiles`) et le mapping
# fraction ↔ out-dir est dans `spatial_datacurve/manifest.json`.
#
# Soumission : sbatch spatial_datacurve/slurm_datacurve_spatial_v2.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_spatial_v2
#SBATCH --array=0-6
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=logs/lora_spatial_v2_%A_%a.out
#SBATCH --error=logs/lora_spatial_v2_%A_%a.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/dinov3_lora_datacurve_r8a16.yaml"   # r=8, alpha=16 (=2r), Q,V — tag r8a16
CODE_DIR="$HOME/benchmark-memoire"                  # clone du dépôt sur Narval (git pull requis)
VENV="$HOME/ENV/bin/activate"
OUT_ROOT="$SCRATCH/sota_screening/lora_spatial_v2"  # nouveau : ne touche pas sota_screening/lora

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

# 7 niveaux de la courbe spatiale (cibles) et répertoires de splits correspondants
FRACS=(0.01 0.05 0.10 0.25 0.50 0.75 1.00)
DIRS=(frac001 frac005 frac010 frac025 frac050 frac075 frac100)
FRAC="${FRACS[$SLURM_ARRAY_TASK_ID]}"
FRAC_DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
PCT=$(python3 -c "print(int(round(${FRAC}*100)))")

echo "═══════════════════════════════════════════════"
echo "LoRA ViT-B DINOv3 SPATIAL v2 | array=$SLURM_ARRAY_TASK_ID → cible=$FRAC (${PCT}%)"
echo "Nœud : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "OUT_ROOT : $OUT_ROOT/$FRAC_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -d "$CODE_DIR/spatial_datacurve/splits" ]] && {
    echo "[ERROR] $CODE_DIR/spatial_datacurve/splits introuvable — git pull requis"
    echo "        (ou exécuter spatial_datacurve/make_spatial_datacurve.py)"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Tuiles (orthomosaïques) → $SLURM_TMPDIR, comme dans slurm_datacurve_lora.sh ──
# Les filepaths des CSV (`arctic_vegetation/<ortho>/<classe>/tile_XXXX.png`) sont
# identiques à ceux du split canonique : la résolution tuiles ←→ fichier est inchangée.
echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles extraites."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

OUT_DIR="$OUT_ROOT/$FRAC_DIR"
mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# 3 seeds séquentiels sur le même GPU
for SEED in 0 1 2; do
    CSV_DIR="$CODE_DIR/spatial_datacurve/splits/${FRAC_DIR}_seed${SEED}"
    [[ ! -f "$CSV_DIR/train.csv" ]] && {
        echo "[ERROR] split spatial absent: $CSV_DIR/train.csv"; exit 1; }

    # Config override générée : fusionne la config LoRA sur base.yaml et redirige
    # csv_dir vers le split SPATIAL de ce (fraction, seed). val/test du même dossier
    # sont inchangés (copies du split canonique). tiles_dir reste ${SLURM_TMPDIR}/tiles
    # (deep-merge src/config.py : seules les clés du bloc paths_narval sont remplacées).
    CFG_OVERRIDE="$SLURM_TMPDIR/cfg_spatial_v2_${FRAC_DIR}_seed${SEED}.yaml"
    cat "$CONFIG" > "$CFG_OVERRIDE"
    cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_datacurve_spatial_v2.sh ──────────────────────────
# Split SPATIAL v2 (orthomosaïques entières / blocs contigus) pour la courbe de données.
paths_narval:
  csv_dir: ${CSV_DIR}
EOF

    # Copie horodatée du config réellement utilisé (traçabilité, cf. slurm_datacurve_lora.sh)
    cp "$CFG_OVERRIDE" "$OUT_DIR/config_used_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_seed${SEED}.yaml"

    echo ""
    echo "─── LoRA SPATIAL v2 cible=$FRAC (${PCT}%) seed=$SEED → $CSV_DIR ───"

    python scripts/datacurve_one_run.py \
        --config "$CFG_OVERRIDE" \
        --fraction 1.0 \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] cible=$FRAC seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] Tâche $SLURM_ARRAY_TASK_ID (cible=$FRAC, ${PCT}%) terminée."
echo "  Résultats : $OUT_DIR/runs/     (tag interne r8a16_frac100_seed{N}, cf. NB en tête)"
echo "  Embeddings : $OUT_DIR/embeddings/"
