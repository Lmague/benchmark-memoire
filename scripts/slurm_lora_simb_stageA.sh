#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# STAGE A — Exploration large LoRA/PEFT — SimDINOv2 ViT-B/16, 3 SEEDS.
# Pipeline A→B→C (2026-09-07) : exploration large → analyse commune → Stage B
# spécialisée → fusion (merge) du plus prometteur.
#
# 13 bras × 3 seeds = 39 runs ≈ 58 GPU-h sur slices A100 MIG 3g.20gb.
# Référence gratuite : canonique r8a8 TOUS blocs Q/V = 0.4781 ± 0.0028 (3 seeds,
# results/all_models_canonical_merged.json — NON re-entraîné ici).
#
# Logique d'isolation (2026-09-07, rev. 2) : les bras POSITION seuls restreignent
# les blocs ; TOUS les autres bras (rang, α, type) tournent sur TOUS les blocs —
# restreindre les couches est déjà un test en soi, on ne le superpose pas aux axes.
#
#   t0  r8a8_b611     — POSITION : blocs 6-11 (hypothèse DINOv3 : b611 ≥ tout)
#   t1  r8a8_b05      — POSITION : contrôle blocs 0-5 (DINOv3 : niveau Full)
#   t2  r8a8_b911     — POSITION : 3 derniers blocs (budget/4)
#   t3  r2a2 (all)    — RANG : r=2, scaling 1
#   t4  r4a4 (all)    — RANG : r=4 (DINOv3 : plateau r2-r8)
#   t5  r16a16 (all)  — RANG : r=16 (DINOv3 : décroche à α=2r — scaling ?)
#   t6  r32a32 (all)  — RANG : r=32 (idem)
#   t7  r8a16 (all)   — ALPHA : scaling 2 à r=8 (vs canonique r8a8)
#   t8  r16a32 (all)  — ALPHA : scaling 2 à r=16 — l'artefact du rang DINOv3 est-il
#                       un artefact de scaling ? (rsLoRA : sous-scaling aux grands r ;
#                       PAS dans la biblio — tentative library_add 2026-09-07, 2 fausses
#                       correspondances, à ajouter à la main si pertinent)
#   t9  r8_rslora     — rsLoRA : scaling √8 = 2.83 (α = r^{3/2} = 22.627, tag r8a22)
#                       [r8 : échelle de scaling 1 / 2 / 2.83]
#   t10 r16_rslora    — rsLoRA : scaling √16 = 4 (α = r^{3/2} = 64, tag r16a64)
#                       [r16 : échelle de scaling 1 / 2 / 4 — si s=4 rattrape r8, le
#                       décrochage DINOv3 était un artefact de scaling, pas de rang]
#   t11 r8a8_qkv      — TYPE : Q+K+V (OUT_DIR SÉPARÉ : target_modules absent du tag)
#   t12 norm_tuning   — PEFT minimal : LayerNorms + head (~0.03 % params, sans LoRA)
#                       (régime ajouté à src/models.py le 2026-09-07 ; réf. DEFLECT
#                       arXiv 2504.17397 : NormTuning ≈ oracle à 0.03 % sur 5 tâches RS)
#
# LECTURE (règle §4.4) : moyennes ± std sur 3 seeds ; tout Δ < 0.005 = ex æquo ;
# > 0.005 = gagnant. Puis analyse commune → Stage B sur mesure
# (scripts/gen_simb_lora_grid.py + slurm_lora_simb_stageB.sh) → Stage C (fusion).
#
# Durée : ~1h20-1h30/seed → 3 seeds ≈ 4h30/tâche → --time=8:00:00.
# Sorties : $SCRATCH/sota_screening/lora_simb_ablation[/qkv]/
#   runs/   simdinov2_vitb16_lora_r{R}a{A}[_b{blocs}]_frac100_seed{N}/  (ou _norm_tuning_)
#   checkpoints/, embeddings/, config_used_*.yaml
#
# PRÉ-REQUIS : $SCRATCH/tiles.zip ; $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth ;
# vendors/sslplant/ ; git pull à jour (incident 2026-08-30 : script non poussé = job à vide).
#
# Soumission : sbatch scripts/slurm_lora_simb_stageA.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_simb_sA
#SBATCH --array=0-12
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=logs/lora_simb_sA_%A_%a.out
#SBATCH --error=logs/lora_simb_sA_%A_%a.err
#SBATCH --account=def-bouguess_gpu

VARIANTS=("r8a8_b611" "r8a8_b05" "r8a8_b911" "r2a2" "r4a4" "r16a16" "r32a32" "r8a16" "r16a32" "r8_rslora" "r16_rslora" "r8a8_qkv" "norm_tuning")

# Tâche hors liste → no-op propre
if [[ $SLURM_ARRAY_TASK_ID -ge ${#VARIANTS[@]} ]]; then
    echo "[slurm] task $SLURM_ARRAY_TASK_ID hors liste — exit."; exit 0
fi

V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
if [[ "$V" == "norm_tuning" ]]; then
    CONFIG="configs/simdinov2_vitb16_norm.yaml"
else
    CONFIG="configs/simdinov2_vitb16_lora_${V}.yaml"
fi
FRAC=1.00
SEEDS="${SEEDS:-0 1 2}"

# Bras QKV → OUT_DIR séparé (collision de tag : target_modules absent du tag)
BASE_OUT="$SCRATCH/sota_screening/lora_simb_ablation"
if [[ "$V" == *"_qkv" ]]; then OUT_DIR="${BASE_OUT}_qkv"; else OUT_DIR="$BASE_OUT"; fi

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "STAGE A — SimDINOv2-B | bras=$V (task $SLURM_ARRAY_TASK_ID/$(( ${#VARIANTS[@]} - 1 )))"
echo "Config  : $CONFIG | seeds : $SEEDS"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -f "$CONFIG" ]] && { echo "[ERROR] config absent: $CONFIG"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Préflight : construction du modèle + comptage params (échec rapide < 5 min) ──
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
counts = {g: sum(p.numel() for p in ps) for g, ps in groups.items()}
print(f"[preflight] OK  groupes={counts}  regime={cfg.regime}  "
      f"r={getattr(cfg.lora, 'r', None)}  alpha={getattr(cfg.lora, 'alpha', None)}  "
      f"targets={tuple(getattr(cfg.lora, 'target_modules', []) or [])}  "
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
echo "[slurm] STAGE A task $SLURM_ARRAY_TASK_ID ($V) terminée."
echo "  Résultats : $OUT_DIR/runs/"
