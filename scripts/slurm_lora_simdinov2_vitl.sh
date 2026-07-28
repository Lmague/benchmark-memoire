#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LoRA r=8 — SimDINOv2 ViT-L/16 × 3 seeds (job unique, pas d'array)
#
# Anciennement task 1 de scripts/slurm_lora_vitl.sh (array 0-1 partagé avec
# DINOv3 ViT-L). Séparé en script indépendant : les deux modèles écrivaient
# EN PARALLÈLE (deux tâches d'array lancées simultanément) vers le même
# --out-dir, et jusqu'au fix de scripts/datacurve_one_run.py, run_dir n'était
# scopé que par fraction+seed (pas par modèle) → collision de chemin, la tâche
# la plus lente écrasant le résultat de l'autre. Perte réelle constatée le
# 2026-07-27/28 : SimDINOv2 ViT-L seed=0 (~4h47 GPU, f1_macro_pres_test=0.4808)
# écrasé par DINOv3-L (terminé après, f1_macro_pres_test=0.4856) — résultat
# irrémédiablement perdu, à refaire (ce script le relance).
# Le fix scope désormais run_dir par ckpt_tag (inclut le nom du modèle), donc
# partager --out-dir est de nouveau sûr, mais on garde deux scripts séparés
# pour éviter tout --time partagé inadapté (durées mesurées très différentes,
# cf. ci-dessous) et pour qu'un seul modèle écrive à la fois par job.
#
# Checkpoints → $SCRATCH/sota_screening/lora_3models/checkpoints/
# Embeddings  → $SCRATCH/sota_screening/lora_3models/embeddings/
# Runs        → $SCRATCH/sota_screening/lora_3models/runs/simdinov2_vitl16_lora_frac100_seed{0,1,2}/
#
# PRÉ-REQUIS :
#   - $SCRATCH/checkpoints/simdinov2_vitl_inat21plantae.pth
#
# --time=16:00:00 : mesuré sur Narval, job 66522732 task 1, seed=0 = 286.8 min
# (~4h47, cf. logs/lora_vitl_66522732_1.out). 3 seeds séquentiels ≈ 14h20 → 16h
# de marge de sécurité.
#
# 20 Go (a100_3g.20gb/a100_4g.20gb) = plafond MIG mesuré (`sinfo -o "%P %G" | grep a100`,
# 2026-07-27) : pas de profil >20 Go en MIG sur Narval. 3g.20gb choisi plutôt que 4g.20gb
# (même mémoire, mais 3-4 instances/nœud contre 1 seule → plus rapide à obtenir).
# OOM du 2026-07-26 causé par batch_size=128 (défaut hérité du ViT-B, ~19,7-22 GiB de
# pic estimé) — corrigé à batch_size=64 dans les configs (~10,5-11 GiB, tient dans
# les 19.62 GiB utilisables du slice). Si indisponible malgré tout, replier sur
# `--gres=gpu:a100:1` + `--mem=60G` (A100 complet, 40 Go, queue plus lente).
#
# Soumission : sbatch scripts/slurm_lora_simdinov2_vitl.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lora_simdinov2_vitl
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/lora_simdinov2_vitl_%j.out
#SBATCH --error=logs/lora_simdinov2_vitl_%j.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/simdinov2_vitl16_lora.yaml"
MODEL="simdinov2_vitl16"
FRAC=1.00

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/lora_3models"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "LoRA r=8 | $MODEL"
echo "Config  : $CONFIG"
echo "Nœud    : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# Tuiles → $SLURM_TMPDIR
echo "[slurm] Extraction tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles extraites."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable"; exit 1
fi

mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# 3 seeds séquentielles — run_dir scopé par ckpt_tag (inclut $MODEL), donc sûr même
# si un autre job (DINOv3 ViT-L) écrit en parallèle vers le même $OUT_DIR.
for SEED in 0 1 2; do
    echo ""
    echo "─── $MODEL seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] $MODEL seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] $MODEL terminé."
echo "  Résultats   : $OUT_DIR/runs/${MODEL}_lora_frac100_seed{0,1,2}/"
echo "  Embeddings  : $OUT_DIR/embeddings/"
