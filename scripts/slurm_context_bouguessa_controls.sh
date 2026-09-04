#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Contrôles Bouguessa (mail début sept. 2026) — points 1 et 2 — JOB CPU PUR.
#
# (1) contexte SEUL (sans l'embedding de la tuile centrale) : sonde canonique sur
#     les colonnes 768:1536 de la fusion R2 [tuile ; contexte].
# (2) contexte PERMUTÉ : sonde sur la fusion où le bloc contexte est réapparié au
#     hasard (5 répétitions, graine 1000+p, permutation indépendante par split) —
#     isole le gain « information spatiale associée à la tuile » de l'effet
#     « simple concaténation d'une seconde représentation ».
# + témoins : fused (reproduction metrics.json 0.5098/0.5070/0.5071) et tile
#   (0:768, baseline tuile-seule par seed).
#
# AUCUN GPU requis : sonde linéaire (lbfgs multinomial, BLAS mono-thread) sur les
# embeddings DÉJÀ EXTRAITS par le job slurm_context_distill_extract_sig.sh —
# $SCRATCH/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed{0,1,2}/
# Parallélisation par PROCESSUS (4 workers × 1 thread BLAS, jamais de threads BLAS
# — AGENTS.md §4.8 : les chiffres canoniques exigent le mono-thread).
#
# Repartable : le script Python saute toute sonde dont le JSON de sortie existe
# déjà ($SCRATCH/context_distill/controls_bouguessa/) — resoumettre librement.
#
# Rapatriement (après le job) :
#   scp -r narval:$SCRATCH/context_distill/controls_bouguessa results/context_distill/
#
# Soumission : git pull puis sbatch scripts/slurm_context_bouguessa_controls.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_bouguessa_ctrl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/context_bouguessa_controls_%j.out
#SBATCH --error=logs/context_bouguessa_controls_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
OUT_DIR="$SCRATCH/context_distill"

echo "═══════════════════════════════════════════════"
echo "ctx_bouguessa_controls (CPU) | Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "SIG_DIR : $SIG_DIR"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR (git pull ?)"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }
for S in 0 1 2; do
    TAG="dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed${S}"
    [[ ! -f "$SIG_DIR/$TAG/test.npy" ]] && {
        echo "[ERROR] $SIG_DIR/$TAG/test.npy absent — le job d'extraction sig ("
        echo "        slurm_context_distill_extract_sig.sh) doit avoir tourné avant."; exit 1; }
done

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs

python scripts/context_bouguessa_controls.py \
    --sig-dir "$SIG_DIR" \
    --out-dir "$OUT_DIR" \
    --workers 4

echo ""
echo "[slurm] contrôles terminés → $OUT_DIR/controls_bouguessa/"
ls -la "$OUT_DIR/controls_bouguessa/" 2>/dev/null
echo ""
echo "[slurm] Rapatriement (depuis la machine locale) :"
echo "  scp -r narval:\$SCRATCH/context_distill/controls_bouguessa results/context_distill/"
