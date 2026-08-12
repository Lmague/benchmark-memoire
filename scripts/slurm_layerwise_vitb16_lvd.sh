#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Géométrie layerwise DINOv3 ViT-B/16 LVD — extraction + probe par couche.
#
# Complète le manque constaté le 2026-08-12 : la géométrie par couche (RankMe,
# anisotropie, F1 du linear probe, gain vs bloc précédent, CKA) n'existait que pour
# vitl16_lvd / vitl16_sat / vitb16_imagenet — pas pour dinov3_vitb16_lvd (le modèle
# de l'ablation LoRA). Ce job produit : embeddings {key}_{split}_layer{idx}.npy +
# results/layerwise_probe.json (F1/geom/CKA par bloc) pour val et test.
#
# Coût estimé : ~1h-GPU A100 (extraction) + ~10-20 min CPU (probe).
# Compte SLURM : def-bouguess_gpu
# Soumission : sbatch scripts/slurm_layerwise_vitb16_lvd.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=lw_vitb16_lvd
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/lw_vitb16_lvd_%j.out
#SBATCH --error=logs/lw_vitb16_lvd_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/Documents/Mémoire"
VENV="$HOME/ENV/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "Layerwise DINOv3-B-LVD | nœud=$SLURMD_NODENAME"

[ ! -d "$CODE_DIR" ] && { echo "ERROR: CODE_DIR absent"; exit 1; }
[ ! -f "$VENV" ] && { echo "ERROR: venv absent"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9 2>/dev/null || true
source "$VENV"
cd "$CODE_DIR"

if [ -f "$SCRATCH/tiles.zip" ]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles extraites."
else
    echo "ERROR: $SCRATCH/tiles.zip absent"; exit 1
fi

mkdir -p "$CODE_DIR/logs" "$CODE_DIR/results/figures"

# 1. Extraction layerwise (val + test ; train inutile pour le classement des couches)
python extract.py --config configs/frozen_dinov3_lvd.yaml \
    --layerwise --splits val test
# NB : sur Narval, emb_dir = ${SCRATCH}/embeddings (auto-détecté) → recopier ensuite
# les dinov3_vitb16_lvd_{val,test}_layer*.npy vers le dépôt local embeddings/.

# 2. Probe + géométrie + CKA par couche (CPU, mono-thread BLAS)
OMP_NUM_THREADS=1 python scripts/layerwise_probe.py \
    --config configs/frozen_eval.yaml \
    --models dinov3_vitb16_lvd \
    --output results/layerwise_probe_vitb16_lvd.json \
    --fig results/figures/layerwise_vitb16_lvd.png

echo "[done] Layerwise DINOv3-B-LVD terminé."
