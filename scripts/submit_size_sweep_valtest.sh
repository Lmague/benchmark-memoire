#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Finalise le sweep des tailles de contexte : zip crops val/test → scp → sbatch.
#
# Pré-requis (déjà en cours/fait par l'agent) :
#   - out/context/context_512/ et out/context/context_2048/  (val+test, crops générés
#     par context_crop.py depuis les COG du SSD) — vérifiés complets avant zip.
#   - Connexion SSH Narval opérationnelle (MFA Duo à valider manuellement ici).
#
# Étapes :
#   1. Vérifie que les counts val+test attendus sont présents (13209+17598 = 30807
#      PNG par taille ; le crop écrit d'ABORD tout 512 puis tout 2048 — on attend
#      les DEUX à 30807).
#   2. Zip depuis out/context/ : context_512_valtest.zip, context_2048_valtest.zip
#      (racine context_512/ + context_2048/ — même convention que les zips Narval).
#   3. scp les 2 zips vers $SCRATCH/ (narval:/scratch/lmague/).
#   4. sbatch scripts/slurm_context_size_sweep.sh   (le job unique : extraction GPU
#      + probes CPU pour les 3 tailles ; il merge train+val/test automatiquement).
#
# Usage : bash scripts/submit_size_sweep_valtest.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -u
EXPECT=30807  # val 13209 + test 17598
OUT="out/context"
NARVAL_DIR="/scratch/lmague"

echo "═══════════════════════════════════════════════"
echo "finalisation sweep tailles | $(date '+%F %T')"
echo "═══════════════════════════════════════════════"

# 1. Vérification des crops
for S in 512 2048; do
  N=$(find "$OUT/context_${S}" -name "*.png" 2>/dev/null | wc -l)
  echo "context_${S}: $N PNG (attendu ≈ $EXPECT)"
  if [[ "$N" -lt "$EXPECT" ]]; then
    echo "[ERROR] context_${S} incomplet ($N < $EXPECT) — le crop tourne encore ? Réessaie plus tard." >&2
    exit 1
  fi
done

# 2. Zip (depuis out/context pour avoir la racine context_<size>/)
echo ""
echo "── zip des crops val/test ──"
for S in 512 2048; do
  (cd "$OUT" && zip -qr "../../context_${S}_valtest.zip" "context_${S}") \
    && echo "context_${S}_valtest.zip : $(ls -la "context_${S}_valtest.zip" | awk '{print $5}') octets"
done

# 3. scp vers Narval
echo ""
echo "── scp vers Narval (valide le MFA Duo) ──"
scp -q "context_512_valtest.zip" "context_2048_valtest.zip" "narval:${NARVAL_DIR}/" \
  && echo "zips envoyés vers narval:${NARVAL_DIR}/"

# 4. sbatch du job unique
echo ""
echo "── sbatch ──"
ssh narval "cd ~/benchmark-memoire && git pull --ff-only && sbatch scripts/slurm_context_size_sweep.sh"

echo ""
echo "Terminé. Surveille : squeue -u lmague ; logs : ~/benchmark-memoire/logs/context_size_sweep_*.out"