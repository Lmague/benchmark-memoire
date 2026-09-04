#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Probes CPU du sweep des tailles de contexte (512/1024/2048) — pallier FROZEN.
#
# Se lance APRÈS le job d'extraction GPU (scripts/slurm_context_size_sweep.sh),
# qui produit :
#   $SCRATCH/context_distill/sig_embeddings/dinov3_vitb16_lvd_FROZEN_fused_ctx<size>_frac100_seed0/
#
# Pour chaque taille : 5 probes canoniques via scripts/context_bouguessa_controls.py
# (même machinerie que les contrôles R2 — BLAS mono-thread, parallélisation par
# processus, repartable par JSON) :
#   - fused          (1536) : courbe d'apport d'information vs échelle de contexte ;
#   - tile           (768)  : témoin (doit ≈ canonique gelé 0,4712) ;
#   - ctx            (768)  : contexte SEUL — ce que chaque échelle porte à elle seule ;
#   - fused_ctxperm{0,1,2}  : contexte réapparié au hasard — l'effet de la simple
#     concaténation, à soustraire du gain pour isoler l'information spatiale associée.
#
# Sorties : $SCRATCH/context_distill/controls_bouguessa/frozen_ctx<size>_seed0_*.json
# Rapatriement : scp -r narval:$SCRATCH/context_distill/controls_bouguessa results/context_distill/
#
# Soumission : git pull puis sbatch scripts/slurm_context_size_sweep_probes.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ctx_size_probes
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/context_size_sweep_probes_%j.out
#SBATCH --error=logs/context_size_sweep_probes_%j.err
#SBATCH --account=def-bouguess_gpu

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
SIG_DIR="$SCRATCH/context_distill/sig_embeddings"
OUT_DIR="$SCRATCH/context_distill"
SIZES="${1:-512 1024 2048}"

echo "═══════════════════════════════════════════════"
echo "ctx_size_probes (CPU) | tailles : $SIZES | Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "═══════════════════════════════════════════════"

[[ ! -d "$CODE_DIR" ]] && { echo "[ERROR] CODE_DIR absent: $CODE_DIR (git pull ?)"; exit 1; }
[[ ! -f "$VENV" ]] && { echo "[ERROR] venv absent: $VENV"; exit 1; }

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs

for SIZE in $SIZES; do
    TAG="dinov3_vitb16_lvd_FROZEN_fused_ctx${SIZE}_frac100_seed0"
    if [[ ! -f "$SIG_DIR/$TAG/test.npy" ]]; then
        echo "[WARN] $SIG_DIR/$TAG/test.npy absent — extraction GPU ("
        echo "        slurm_context_size_sweep.sh) pas encore faite pour ${SIZE}px — taille sautée" >&2
        continue
    fi
    echo ""
    echo "─── probes frozen contexte ${SIZE}px ───"
    python scripts/context_bouguessa_controls.py \
        --sig-dir "$SIG_DIR" \
        --tag "$TAG" \
        --seeds 0 \
        --perm-reps 3 \
        --json-prefix "frozen_ctx${SIZE}" \
        --out-dir "$OUT_DIR" \
        --workers 4
    [[ $? -ne 0 ]] && echo "[WARN] probes ${SIZE}px échoués" >&2
done

echo ""
echo "[slurm] probes terminés → $OUT_DIR/controls_bouguessa/"
ls -la "$OUT_DIR/controls_bouguessa/" 2>/dev/null
echo ""
echo "[slurm] Rapatriement (depuis la machine locale) :"
echo "  scp -r narval:\$SCRATCH/context_distill/controls_bouguessa results/context_distill/"
