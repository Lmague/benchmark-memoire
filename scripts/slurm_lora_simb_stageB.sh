#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# STAGE B — Grille r × α à la position GAGNANTE de la Stage A — SimDINOv2 ViT-B/16,
# 3 seeds. Pipeline A→B→C (2026-09-07) : position → grille r×α → consolidation.
#
# PRÉ-REQUIS :
#   1. Stage A terminée → position gagnante notée (règle de décision dans son en-tête)
#   2. Configs générées :  python scripts/gen_simb_lora_grid.py --position <gagnant> [--qkv]
#   3. git pull à jour sur Narval.
#
# Grille par défaut (position b611) : r ∈ {2,4,8,16} × α ∈ {r, 2r} = 8 configs × 3 seeds.
# Seed 0 de r8a8_b611 est déjà fait (Stage A) → --skip-if-done le réutilise (24 runs
# demandés, 22 calculés). Si le gagnant est b05 ou b911, passer la liste via --export.
#
# Bras QKV (type) : tag IDENTIQUE au QV (target_modules absent du tag) → OUT_DIR _qkv.
# Le script le détecte automatiquement via le suffixe _qkv du nom de variante.
#
# Soumission (exemples) :
#   sbatch scripts/slurm_lora_simb_stageB.sh                        # grille b611 par défaut
#   sbatch --export=ALL,VARIANTS="r8a8_b05 r8a16_b05 r2a2_b05 r16a16_b05" \
#          --array=0-3 scripts/slurm_lora_simb_stageB.sh            # gagnant b05, sous-grille
#   sbatch --export=ALL,VARIANTS="r8a8_b611_qkv" --array=0-0 \
#          scripts/slurm_lora_simb_stageB.sh                        # bras QKV seul
#
# Tâches au-delà de la liste → exit 0 immédiat (array 0-7 soumis tel quel OK).
# Durée : ~1h20-1h30/seed → 3 seeds ≈ 4h30/tâche → --time=8:00:00.
# Sorties : mêmes OUT_DIR que la Stage A (réutilisation des seeds via skip-if-done) :
#   $SCRATCH/sota_screening/lora_simb_ablation[/runs|checkpoints|embeddings]
#   $SCRATCH/sota_screening/lora_simb_ablation_qkv (bras QKV)
#
# APRÈS (STAGE C) : rapatrier runs+embeddings, F1 canonique sur les embeddings
# (protocole scripts/probe_lora3_new_runs.py, BLAS mono-thread §4.8), fusion dans
# results/all_models_canonical_merged.json, géométrie par-seed si le gagnant change.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_simb_sB
#SBATCH --array=0-7
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_simb_sB_%A_%a.out
#SBATCH --error=logs/lora_simb_sB_%A_%a.err
#SBATCH --account=def-bouguess_gpu

# Liste des variantes (surchargeable : sbatch --export=ALL,VARIANTS="...")
DEFAULT_VARIANTS="r2a2_b611 r2a4_b611 r4a4_b611 r4a8_b611 r8a8_b611 r8a16_b611 r16a16_b611 r16a32_b611"
read -ra VARIANTS <<< "${VARIANTS:-$DEFAULT_VARIANTS}"

FRAC=1.00
SEEDS="${SEEDS:-0 1 2}"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
BASE_OUT="$SCRATCH/sota_screening/lora_simb_ablation"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

# Tâche hors liste → no-op propre (array 0-7 soumissible tel quel)
if [[ $SLURM_ARRAY_TASK_ID -ge ${#VARIANTS[@]} ]]; then
    echo "[slurm] task $SLURM_ARRAY_TASK_ID hors liste (${#VARIANTS[@]} variantes) — exit."
    exit 0
fi

V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="configs/simdinov2_vitb16_lora_${V}.yaml"

# Bras QKV → OUT_DIR séparé (collision de tag : target_modules absent du tag)
if [[ "$V" == *"_qkv" ]]; then
    OUT_DIR="${BASE_OUT}_qkv"
else
    OUT_DIR="$BASE_OUT"
fi

echo "═══════════════════════════════════════════════"
echo "STAGE B — grille r×α SimDINOv2-B | $V (task $SLURM_ARRAY_TASK_ID)"
echo "Config  : $CONFIG | seeds : $SEEDS"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -f "$CONFIG" ]] && { echo "[ERROR] config absent: $CONFIG — lancer d'abord \
scripts/gen_simb_lora_grid.py --position <gagnant>"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Préflight (échec rapide si l'injection diffère de la Stage A) ──
python - "$CONFIG" <<'PYEOF' || { echo "[ERROR] préflight échoué — job stoppé"; exit 1; }
import os, sys
from src.config import load_config
from src.models import build_model

cfg = load_config(sys.argv[1])
ckpt = cfg.raw.get("checkpoint")
if ckpt and not os.path.isabs(ckpt):
    ckpt = os.path.join(cfg.paths.ckpt_dir, ckpt)
assert ckpt and os.path.isfile(ckpt), f"checkpoint introuvable : {ckpt}"

model, groups = build_model(cfg.model.name, cfg.regime, cfg.model.num_classes,
                            lora=cfg.lora, checkpoint=ckpt)
n_lora = sum(p.numel() for p in groups.get("lora", []))
print(f"[preflight] OK  lora={n_lora:,}  r={cfg.lora.r}  alpha={cfg.lora.alpha}  "
      f"targets={tuple(cfg.lora.target_modules)}  "
      f"blocks={getattr(cfg.lora, 'lora_block_indices', None) or 'TOUS'}")
PYEOF

# ── Tuiles → $SLURM_TMPDIR ──
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles extraites."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"
cp "$CONFIG" "$OUT_DIR/config_used_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${V}.yaml"

for SEED in $SEEDS; do
    echo ""
    echo "─── $V seed=$SEED ───"
    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done
    [[ $? -ne 0 ]] && echo "[ERROR] $V seed=$SEED échoué — continuation" >&2
done

echo ""
echo "[slurm] STAGE B task $SLURM_ARRAY_TASK_ID ($V) terminée."
echo "  Résultats : $OUT_DIR/runs/ (tags _r{R}a{A}[_b{blocs}])"
