#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Self-distillation contexte→tuile — DINOv3-B LoRA student, 3 axes de variation.
#
# 1 GPU A100 40 Go COMPLET (--gres=gpu:a100:1, PAS de nœud entier, PAS de slice MIG —
# le teacher (forward seul, gelé, pas d'état optimizer) + le student ViT-B LoRA (peu
# de params entraînables) tiennent sur un A100 complet mais dépassent les slices MIG
# 10-20 Go utilisées par les autres jobs LoRA de ce dépôt).
#
# ⚠️ --time NON MESURÉ (aucun run GPU exécuté au moment d'écrire ce script). Estimation
# prudente extrapolée de slurm_lora_dinov3_vitl.sh (LoRA r8 ViT-L seul = 356.5 min/seed
# mesuré, job 66522732) + surcoût forward teacher à chaque step + I/O tuile+contexte
# doublé → --time=24:00:00. Design B ajoute un 3e forward (student sur le contexte, en
# plus de student-sur-tuile et teacher-sur-contexte) — surcoût non mesuré, même budget
# de temps par prudence. À AJUSTER après le premier run réel (logs/context_distill_*.out).
#
# ⚠️ --mem UNIFORME (64G) quels que soient contexte/design/teacher : context_crop.py
# REDIMENSIONNE tout contexte à out-size=224 AVANT sauvegarde (cf. docstring de
# context_crop.py) — le tenseur chargé en entraînement a la MÊME taille peu importe
# --context-size. Design B fait un forward de plus (student sur le contexte) mais avec
# le MÊME petit modèle LoRA déjà chargé, surcoût mémoire marginal attendu (non mesuré).
#
# Usage : $1 = seed (0, 1 ou 2, obligatoire). $2 = context_size (512|1024|2048, défaut
# 1024). $3 = design (A|B, défaut A). $4 = teacher (nom d'un backbone frozen, défaut
# dinov3_vitl16_lvd ; ou 'ema_self' pour le momentum-teacher self-distillation).
# Un job par seed (pas de boucle interne sur les 3 seeds) — chaque run est plus coûteux
# que les jobs LoRA habituels (forward teacher/contexte supplémentaire à chaque step),
# la parallélisation inter-seeds raccourcit le temps d'horloge total.
#
# Ordre recommandé (cf. scripts/context_distill_README.md) :
#   1. R1 (1024px, teacher DINOv3-L, Design A — l'expérience principale) :
#        sbatch scripts/slurm_context_distill.sh 0
#        sbatch scripts/slurm_context_distill.sh 1
#        sbatch scripts/slurm_context_distill.sh 2
#   2. SI R1 bat la baseline spatiale (F1=0.4827±0.0042, frac100 LoRA r=8, mesuré
#      2026-08-29 depuis results/spatial_datacurve_CANONICAL.csv, cf. README §3) :
#      R2 (contexte 512px) et R3 (contexte 2048px), même teacher/design :
#        sbatch scripts/slurm_context_distill.sh 0 512
#        sbatch scripts/slurm_context_distill.sh 1 512
#        sbatch scripts/slurm_context_distill.sh 2 512
#        sbatch scripts/slurm_context_distill.sh 0 2048   # (+ seeds 1, 2)
#   3. Contrôles (indépendants de 1-2, mêmes 3 seeds chacun) :
#      a. EMA self-teacher — isole si le gain vient d'un teacher externe plus riche
#         ou juste d'un signal de self-distillation (contexte 1024px, comme R1) :
#           sbatch scripts/slurm_context_distill.sh 0 1024 A ema_self
#           sbatch scripts/slurm_context_distill.sh 1 1024 A ema_self
#           sbatch scripts/slurm_context_distill.sh 2 1024 A ema_self
#      b. Design B — le contexte aide-t-il EN PLUS à l'inférence (pas seulement à
#         l'entraînement) ? PRÉREQUIS SUPPLÉMENTAIRE : le contexte doit exister pour
#         val ET test aussi (pas seulement train) — relancer context_crop.py EN LOCAL
#         sur val.csv/test.csv et retransférer context_1024.zip avant de soumettre :
#           sbatch scripts/slurm_context_distill.sh 0 1024 B
#           sbatch scripts/slurm_context_distill.sh 1 1024 B
#           sbatch scripts/slurm_context_distill.sh 2 1024 B
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip                      (tuiles 224px, convention existante)
#   - $SCRATCH/context_<size>.zip   (produit par scripts/context_crop.py EN LOCAL puis
#     transféré — cf. slurm_context_crop.sh). Design B : ce zip doit AUSSI contenir
#     les crops val/test, pas seulement train (contrairement à R1/R2/R3/EMA).
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitl16-pretrain-lvd1689m/ (sauf si
#     $4=ema_self, auquel cas aucun teacher externe n'est chargé)
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=context_distill
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/context_distill_%j.out
#SBATCH --error=logs/context_distill_%j.err
#SBATCH --account=def-bouguess_gpu

SEED="${1:?usage: sbatch scripts/slurm_context_distill.sh <seed:0|1|2> [context_size:512|1024|2048] [design:A|B] [teacher]}"
CONTEXT_SIZE="${2:-1024}"
DESIGN="${3:-A}"
TEACHER="${4:-dinov3_vitl16_lvd}"
CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
SPATIAL_CSV_DIR="$CODE_DIR/splits_spatial/frac100_seed${SEED}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "context_distill | seed=$SEED context_size=$CONTEXT_SIZE design=$DESIGN teacher=$TEACHER"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -f "$SPATIAL_CSV_DIR/train.csv" ]] && {
    echo "[ERROR] split spatial absent: $SPATIAL_CSV_DIR/train.csv (scripts/make_spatial_splits.py)"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs

echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

CONTEXT_ZIP="$SCRATCH/context_${CONTEXT_SIZE}.zip"
[[ ! -f "$CONTEXT_ZIP" ]] && {
    echo "[ERROR] $CONTEXT_ZIP introuvable — lancer context_crop.py EN LOCAL et transférer"
    echo "        les context_<size>.zip d'abord (cf. scripts/slurm_context_crop.sh)."
    exit 1
}
echo "[slurm] Extraction contexte ($CONTEXT_ZIP) → $SLURM_TMPDIR ..."
unzip -q "$CONTEXT_ZIP" -d "$SLURM_TMPDIR/"
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ ! -d "$CONTEXT_DIR" ]] && { echo "[ERROR] $CONTEXT_DIR absent après unzip"; exit 1; }

# Design B : vérifie que le contexte val EXISTE (pas seulement train) — sinon échec
# tardif et confus au milieu de l'entraînement (FileNotFoundError dans le DataLoader).
if [[ "$DESIGN" == "B" ]]; then
    FIRST_VAL_FP=$(sed -n '2p' "${SPATIAL_CSV_DIR}/val.csv" | cut -d',' -f1)
    if [[ -z "$FIRST_VAL_FP" || ! -f "$CONTEXT_DIR/$FIRST_VAL_FP" ]]; then
        echo "[ERROR] Design B nécessite le contexte pour val ET test, pas seulement train."
        echo "        Fichier attendu introuvable : $CONTEXT_DIR/$FIRST_VAL_FP"
        echo "        Relancer context_crop.py EN LOCAL sur val.csv/test.csv (--out-dir le même"
        echo "        dossier out/context) et retransférer context_${CONTEXT_SIZE}.zip."
        exit 1
    fi
fi

# Override généré : pointe csv_dir vers le split spatial de ce seed ET tiles_dir vers
# le SLURM_TMPDIR (même mécanique que slurm_datacurve_spatial.sh:90-98).
CFG_OVERRIDE="$SLURM_TMPDIR/cfg_context_distill_seed${SEED}_ctx${CONTEXT_SIZE}_d${DESIGN}.yaml"
cat "$CONFIG" > "$CFG_OVERRIDE"
cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_distill.sh ───────────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

mkdir -p "$OUT_DIR"
cp "$CFG_OVERRIDE" "$OUT_DIR/config_used_${SLURM_JOB_ID}_seed${SEED}_ctx${CONTEXT_SIZE}_d${DESIGN}.yaml"

EMA_ARGS=()
if [[ "$TEACHER" == "ema_self" ]]; then
    EMA_ARGS=(--ema-momentum 0.999)
fi

python scripts/context_distill.py \
    --config "$CFG_OVERRIDE" \
    --teacher "$TEACHER" \
    --context-size "$CONTEXT_SIZE" \
    --context-dir "$CONTEXT_DIR" \
    --design "$DESIGN" \
    "${EMA_ARGS[@]}" \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --skip-if-done

echo ""
echo "[slurm] seed=$SEED context_size=$CONTEXT_SIZE design=$DESIGN teacher=$TEACHER terminé."
echo "  Résultats sous : $OUT_DIR/runs/  (tag = {model}_ctxdistill_d${DESIGN}_t*_ctx${CONTEXT_SIZE}_r2a4_frac100_seed${SEED})"
