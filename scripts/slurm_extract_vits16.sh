#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Extraction frozen — DINOv3 ViT-S/16 LVD-1689M (HuggingFace transformers)
#
# Single job GPU : embeddings {val,test,train} → $SCRATCH/embeddings/ sous la
# convention canonique {model}_{split}.npy (emb_dir auto-détecté via base.yaml).
# ViT-S ~22M params : batch 128 (défaut base.yaml) très confortable sur un
# demi-GPU A100 — extraction estimée ~10-20 min (le layerwise ViT-B ≈ 1h).
#
# PRÉ-REQUIS :
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vits16-pretrain-lvd1689m/
#     (cache HF pré-téléchargé — HF_HUB_OFFLINE=1, pas de réseau au run)
#   - $SCRATCH/tiles.zip (tuiles, extraites dans $SLURM_TMPDIR)
#
# APRÈS l'extraction (intégration pipeline canonique) :
#   1. vérifier les .npy : ls $SCRATCH/embeddings/dinov3_vits16_lvd_{val,test,train}.npy
#   2. ajouter `dinov3_vits16_lvd` à configs/probe_all.yaml ET
#      configs/benchmark_12models.yaml (liste `models`)
#   3. relancer le probe canonique (slurm_probe_all.sh / run_pipeline.py) puis
#      mettre à jour all_models_canonical_merged.json (schéma f1_linear_probe,
#      f1_std, f1_seeds, dim=384, ...) — cf. consignes d'intégration.
#
# Soumission : sbatch scripts/slurm_extract_vits16.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=extract_vits16
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/extract_vits16_%j.out
#SBATCH --error=logs/extract_vits16_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR

echo "═══════════════════════════════════════════════"
echo "Extraction frozen | dinov3_vits16_lvd"
echo "Nœud    : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID"
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

mkdir -p "$CODE_DIR/logs"

echo "[extract] lancement extract.py (val + test + train, embeddings → \$SCRATCH/embeddings) ..."
python extract.py --config configs/frozen_dinov3_vits16.yaml
RC=$?

if [[ $RC -ne 0 ]]; then
    echo "[ERROR] extract.py échoué (exit=$RC)" >&2
    exit $RC
fi

echo ""
echo "[slurm] Extraction terminée. Vérifier :"
echo "  ls $SCRATCH/embeddings/dinov3_vits16_lvd_{val,test,train}.npy"
echo "  ls $SCRATCH/embeddings/dinov3_vits16_lvd_{val,test,train}_labels.npy"
echo "[slurm] done."
