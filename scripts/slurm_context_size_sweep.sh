#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Sweep de la TAILLE du contexte (512/1024/2048) — DINOv3-B GELÉ, sonde fusionnée.
#
# Demande de M. Bouguessa : « regarder les différentes tailles de contexte (512,
# 1024, 2048), avant de passer à des modèles DINOv3 plus grands ». Premier pallier
# NON ENTRAÎNÉ (courbe d'apport d'information vs échelle, cf. ANALYSE.md §7) —
# si la courbe le justifie, R2 Design B entraîné à 512/2048 suit en second pallier
# (sbatch scripts/slurm_context_distill.sh <size> B, cf. NEEDS_HUMAN.md).
#
# Pour chaque taille : EXTRACTION SEULE des features frozen fusionnées [tile ; ctx]
# (3 splits) → $SCRATCH/context_distill/sig_embeddings/<tag>_frac100_seed0/.
# Les PROBES (fused / ctx seul / contexte permuté) tournent ensuite dans un JOB CPU
# SÉPARÉ : sbatch scripts/slurm_context_size_sweep_probes.sh (machinerie de
# scripts/context_bouguessa_controls.py) — convention « extraction GPU / probes CPU ».
#
# Sorties :
#   - $SCRATCH/context_distill/sig_embeddings/dinov3_vitb16_lvd_FROZEN_fused_ctx<size>_frac100_seed0/
#   - (après le job CPU) $SCRATCH/context_distill/controls_bouguessa/frozen_ctx<size>_seed0_*.json
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip
#   - $SCRATCH/context_512.zip, context_1024.zip, context_2048.zip  (train crops ;
#     1024 contient déjà val/test) ET, pour 512/2048, les val/test manquants :
#     $SCRATCH/context_512_valtest.zip + context_2048_valtest.zip
#     (produits EN LOCAL par scripts/context_crop.py sur val.csv+test.csv —
#     cf. NEEDS_HUMAN.md section correspondante). Les deux zips sont fusionnés
#     dans le même dossier (aucune collision de noms : tuiles disjointes).
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_size_sweep
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/context_size_sweep_%j.out
#SBATCH --error=logs/context_size_sweep_%j.err
#SBATCH --account=def-bouguess_gpu

CONFIG="configs/context_distill_dinov3b.yaml"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"
SIZES="${1:-512 1024 2048}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ctx_size_sweep | tailles : $SIZES"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

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

for SIZE in $SIZES; do
    echo ""
    echo "═══════ taille de contexte : ${SIZE}px ═══════"
    CONTEXT_DIR="$SLURM_TMPDIR/context_${SIZE}"
    rm -rf "$CONTEXT_DIR"
    if [[ ! -f "$SCRATCH/context_${SIZE}.zip" ]]; then
        echo "[ERROR] $SCRATCH/context_${SIZE}.zip introuvable — taille $SIZE sautée" >&2
        continue
    fi
    unzip -q "$SCRATCH/context_${SIZE}.zip" -d "$SLURM_TMPDIR/"
    # val/test manquants pour 512/2048 : merge du zip complémentaire (tuiles disjointes,
    # aucune collision de noms). 1024 contient déjà val/test → pas de zip _valtest.
    if [[ -f "$SCRATCH/context_${SIZE}_valtest.zip" ]]; then
        echo "[slurm] Merge val/test (${SIZE}px) ..."
        unzip -q "$SCRATCH/context_${SIZE}_valtest.zip" -d "$SLURM_TMPDIR/"
    fi
    N_TILES=$(ls "$CONTEXT_DIR" 2>/dev/null | wc -l)
    echo "[slurm] context_${SIZE} : $N_TILES crops"
    if [[ "$N_TILES" -lt 79000 ]]; then
        echo "[WARN] ${SIZE}px : $N_TILES crops < 79k attendus (49433 train + 13209 val + 17598 test) — vérifier les zips" >&2
    fi

    python scripts/context_size_sweep.py \
        --config "$CONFIG" \
        --context-dir "$CONTEXT_DIR" \
        --context-size "$SIZE" \
        --out-dir "$OUT_DIR" \
        --skip-probes
    [[ $? -ne 0 ]] && echo "[WARN] extraction ${SIZE}px échouée" >&2
done

echo ""
echo "[slurm] extractions terminées → $OUT_DIR/sig_embeddings/"
ls -la "$OUT_DIR/sig_embeddings/" 2>/dev/null | grep FROZEN || true
echo ""
echo "[slurm] Prochaines étapes :"
echo "  1) sbatch scripts/slurm_context_size_sweep_probes.sh   # probes CPU (fused/ctx/perm)"
echo "  2) scp -r narval:\$SCRATCH/context_distill/controls_bouguessa results/context_distill/"
