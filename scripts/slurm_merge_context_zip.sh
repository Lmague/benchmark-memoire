#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# FUSIONNE le zip de contexte train + val/test en un zip COMPLET, côté Narval.
#
# Nécessaire pour les runs Design B (R2) et le sweep : le contexte doit exister
# pour val/test aussi. Les zips <size>.zip actuels ne contiennent que le TRAIN
# (49433 PNG) ; context_<size>_valtest.zip (30807 PNG) doit être fusionné dans le
# même dossier → zip complet ~80240 PNG.
#
# ROBUSTE : conçu pour être lancé en sbatch (survit aux coupures SSH — contraire-
# ment à la commande manuelle qui s'est fait couper en plein zip) et REPARTABLE :
# - gère l'état laissé par une tentative interrompue (context_<size>.zip déjà
#   renommé *_trainonly.zip, zip complet partiel, dossier m<size> résiduel) ;
# - reconstruit TOUJOURS le zip complet depuis zéro (rm avant zip, pas d'append).
# - validation : compte les PNG du zip complet (attendu 80240) avant le swap.
#
# Usage : sbatch scripts/slurm_merge_context_zip.sh 512        (ou 2048 ; 1024
#         est déjà complet et n'a pas de zip _valtest — ne pas lancer pour 1024)
#
# Sortie : $SCRATCH/context_<size>.zip  (complet, 80240 PNG) ; l'ancien train-seul
#          est conservé sous context_<size>_trainonly.zip.
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=merge_ctx_zip
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/merge_context_zip_%j.out
#SBATCH --error=logs/merge_context_zip_%j.err
#SBATCH --account=def-bouguess_gpu

SIZE="${1:-512}"
EXPECT=$((49433 + 30807))   # train + (val 13209 + test 17598) = 80240
cd "$SCRATCH" || exit 1

echo "═══════════════════════════════════════════════"
echo "merge contexte ${SIZE}px | Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"
ls -la "context_${SIZE}"*.zip 2>/dev/null

# Source train : selon l'état laissé par une tentative précédente
TRAIN_ZIP=""
[[ -f "context_${SIZE}.zip" ]]             && TRAIN_ZIP="context_${SIZE}.zip"
[[ -f "context_${SIZE}_trainonly.zip" ]]   && TRAIN_ZIP="context_${SIZE}_trainonly.zip"
VLT_ZIP="context_${SIZE}_valtest.zip"

[[ -z "$TRAIN_ZIP" ]] && { echo "[ERREUR] ni context_${SIZE}.zip ni *_trainonly.zip — rien à merger"; exit 1; }
[[ ! -f "$VLT_ZIP" ]] && { echo "[ERREUR] $VLT_ZIP absent (1024 déjà complet ? ce job est pour 512/2048)"; exit 1; }
echo "[merge] train source : $TRAIN_ZIP | valtest : $VLT_ZIP"

# Reconstruit TOUJOURS le zip complet (une tentative coupée a pu laisser un partiel)
rm -rf "m${SIZE}" "context_${SIZE}_full.zip"
mkdir -p "m${SIZE}"
unzip -qo "$TRAIN_ZIP" -d "m${SIZE}"
unzip -qo "$VLT_ZIP"   -d "m${SIZE}"
(cd "m${SIZE}" && zip -qr "../context_${SIZE}_full.zip" "context_${SIZE}")
rm -rf "m${SIZE}"

# Validation : compte des PNG dans le zip complet
N=$(unzip -l "context_${SIZE}_full.zip" 2>/dev/null | grep -c '\.png$')
echo "[merge] PNG dans le zip complet : $N (attendu $EXPECT)"
if [[ "$N" != "$EXPECT" ]]; then
    echo "[ERREUR] compte inattendu — on ne remplace RIEN. Vérifier les zips source." >&2
    exit 1
fi

# Swap atomique
[[ -f "context_${SIZE}.zip" ]] && mv "context_${SIZE}.zip" "context_${SIZE}_trainonly.zip"
mv "context_${SIZE}_full.zip" "context_${SIZE}.zip"
echo ""
echo "[merge] OK — context_${SIZE}.zip est maintenant complet :"
ls -la "context_${SIZE}.zip" "context_${SIZE}_trainonly.zip" 2>/dev/null
echo ""
echo "Prêt pour : sbatch scripts/slurm_context_distill.sh ${SIZE} B"