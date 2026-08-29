#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Découpe des fenêtres de contexte (512/1024/2048px) pour l'expérience de
# self-distillation contexte→tuile — CPU UNIQUEMENT (rasterio/geopandas, pas de GPU).
#
# ⚠️ CHEMIN PAR DÉFAUT RECOMMANDÉ : exécuter scripts/context_crop.py EN LOCAL, PAS via
# ce script. Les 38 COG bruts (High-Resolution Arctic Vegetation Maps...) ne sont
# PRÉSENTS QUE localement (comme tous les autres jobs Narval de ce dépôt, seul
# `$SCRATCH/tiles.zip` — déjà tuilé 224px — est transféré ; les orthomosaïques brutes,
# probablement des dizaines de Go, n'ont JAMAIS été rsyncées sur Narval). Ce script
# N'EST DONC UTILISABLE QUE SI tu as toi-même transféré les COG bruts sous
# $SCRATCH/<DATASET_DIR_NAME>/ au préalable — sinon il échoue au premier raster
# manquant (erreur explicite de context_crop.py, pas un plantage silencieux).
#
# Local (recommandé, ~15 orthos train, ~10-30 min CPU total mesuré par extrapolation
# de 20230724_alder39_m3m = 28369 fenêtres / 31s, cf. scripts/context_distill_README.md
# §Validation) :
#   python scripts/context_crop.py \
#     --split-csv splits_spatial/frac100_seed0/train.csv \
#     --context-sizes 512,1024,2048 --out-size 224 --out-dir out/context
#   cd out/context && for d in context_*; do zip -qr "../../${d}.zip" "$d"; done
#   scp context_512.zip context_1024.zip context_2048.zip narval:$SCRATCH/
#
# frac100_seed{0,1,2}/train.csv sont BIT-IDENTIQUES (md5 vérifié, 2026-08-29) — à
# 100% de fraction il n'y a pas de sous-échantillonnage stochastique par seed, donc UN
# SEUL passage sur les orthos (n'importe quel seed) couvre les 3 seeds d'entraînement.
#
# Soumission (SI les COG sont sur Narval) : sbatch scripts/slurm_context_crop.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=context_crop
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/context_crop_%j.out
#SBATCH --error=logs/context_crop_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
# Racine du dataset Arctic-TVC BRUT (COG + labels) — À TRANSFÉRER SOI-MÊME sur
# $SCRATCH avant de soumettre ce job (cf. avertissement en tête de fichier).
DATASET_ROOT="$SCRATCH/High-Resolution Arctic Vegetation Maps and Photogrammetry Data from Drone Surveys at Trail Valley Creek, Northwest Territories"
OUT_DIR="$SCRATCH/context_crop_out"

echo "═══════════════════════════════════════════════"
echo "context_crop | Nœud: $SLURMD_NODENAME | Job: $SLURM_JOB_ID"
echo "DATASET_ROOT : $DATASET_ROOT"
echo "OUT_DIR      : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
[[ ! -d "$DATASET_ROOT" ]] && {
    echo "[ERROR] $DATASET_ROOT introuvable — les COG bruts n'ont pas été transférés sur"
    echo "        Narval. Chemin recommandé : lancer context_crop.py EN LOCAL (cf. en-tête"
    echo "        de ce fichier) puis transférer context_<size>.zip, pas les COG."
    exit 1
}

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p "$OUT_DIR" logs

python scripts/context_crop.py \
    --split-csv splits_spatial/frac100_seed0/train.csv \
    --context-sizes 512,1024,2048 \
    --out-size 224 \
    --out-dir "$OUT_DIR" \
    --dataset-root "$DATASET_ROOT"

echo ""
echo "[slurm] Zip des sorties (pour unzip par slurm_context_distill.sh, convention tiles.zip)..."
cd "$OUT_DIR"
for size in 512 1024 2048; do
    d="context_${size}"
    [[ -d "$d" ]] && zip -qr "$SCRATCH/${d}.zip" "$d" && echo "  → $SCRATCH/${d}.zip"
done

echo "[slurm] Terminé. Manifeste : $OUT_DIR/coords_manifest.json"
