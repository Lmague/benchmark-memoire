#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# FT SoTA — SimDINOv2 ViT-B/16 (iNat21 Plantae), régime FULL, 100% des données, 3 seeds.
# Miroir exact de scripts/slurm_sota_screening.sh (array 0, config vitb16_fulft_sota.yaml)
# mais pour le backbone SimDINOv2 — objectif : mesurer si le fine-tuning complet d'un SSL
# déjà fort et domain-adapté (plantes) apporte un gain mesurable par rapport au frozen
# (simdinov2_vitb16, déjà benchmarké), ou sature déjà en frozen.
#
# OUT_DIR dédié (sota_screening/simdinov2_vitb16_full/), DISTINCT de sota_screening/full/
# (réservé à vitb16 ImageNet) — évite toute collision avec les checkpoints/embeddings
# canoniques déjà utilisés dans results/relance2/relance2_11cls.json (F1=0.4708).
#
# PRÉ-REQUIS (vérifiés ci-dessous, non résolus automatiquement) :
#   - $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth doit déjà exister (même
#     fichier que l'extraction frozen existante, configs/frozen_simdinov2_vitb16.yaml).
#   - vendors/sslplant/ (repo ilyassmoummad/sslplant) doit déjà être cloné sous
#     $CODE_DIR/vendors/sslplant — le clone auto (_ensure_sslplant_on_path) nécessite un
#     accès réseau souvent indisponible sur les nœuds de calcul Narval. Si absent, cloner
#     DEPUIS LE LOGIN NODE avant soumission :
#       git clone --depth 1 https://github.com/ilyassmoummad/sslplant.git \
#           $HOME/benchmark-memoire/vendors/sslplant
#
# Soumission (MANUELLE — MFA) :  sbatch scripts/slurm_ft_simdinov2_vitb16_full.sh
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=ft_simdv2b_full
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/ft_simdv2b_full_%j.out
#SBATCH --error=logs/ft_simdv2b_full_%j.err
#SBATCH --account=def-bouguess_gpu

# ─── Configuration ─────────────────────────────────────────────────────────────
CONFIG="configs/ft_simdinov2_vitb16_full.yaml"
REGIME_TAG="simdinov2_vitb16_full"

CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/sota_screening/${REGIME_TAG}"
FRAC=1.00   # screening = 100% des données (pas de data curve pour ce lot)
PRETRAIN_CKPT="$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"
export CODE_DIR
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════"
echo "FT SimDINOv2 ViT-B/16 (iNat Plantae) — régime=full  config=$CONFIG"
echo "Nœud : $SLURMD_NODENAME  | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
echo "═══════════════════════════════════════════════"

# ── Vérifications préliminaires ───────────────────────────────────────────────
if [[ ! -d "$CODE_DIR" ]]; then
    echo "[ERROR] CODE_DIR non trouvé : $CODE_DIR" >&2
    exit 1
fi
if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] venv non trouvé : $VENV" >&2
    exit 1
fi
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
    echo "[ERROR] checkpoint pré-entraîné SimDINOv2 introuvable : $PRETRAIN_CKPT" >&2
    echo "        (même fichier que l'extraction frozen existante — doit déjà être présent)" >&2
    exit 1
fi
if [[ ! -d "$CODE_DIR/vendors/sslplant" ]]; then
    echo "[WARN] $CODE_DIR/vendors/sslplant absent — le clone auto nécessite un accès" >&2
    echo "       réseau, souvent indisponible sur un nœud de calcul Narval. Si le job" >&2
    echo "       échoue au clone, relancer depuis le login node :" >&2
    echo "       git clone --depth 1 https://github.com/ilyassmoummad/sslplant.git $CODE_DIR/vendors/sslplant" >&2
fi

# ── Environnement Python ──────────────────────────────────────────────────────
module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$CODE_DIR"

# ── Copie/extraction des tuiles → $SLURM_TMPDIR (SSD local rapide) ────────────
echo "[slurm] Extraction des tuiles → $SLURM_TMPDIR ..."
if [[ -f "$SCRATCH/tiles.zip" ]]; then
    unzip -q "$SCRATCH/tiles.zip" -d "$SLURM_TMPDIR/"
    echo "[slurm] Extraction terminée : $(find $SLURM_TMPDIR/tiles -name '*.png' | wc -l) tuiles."
else
    echo "[ERROR] $SCRATCH/tiles.zip introuvable" >&2
    exit 1
fi

# ── Création des dossiers de sortie ──────────────────────────────────────────
mkdir -p "$OUT_DIR/runs" "$OUT_DIR/checkpoints" "$OUT_DIR/embeddings"
mkdir -p "$CODE_DIR/logs"

# ── Boucle sur les 3 seeds (séquentielle, même MIG slice) ────────────────────
for SEED in 0 1 2; do
    echo ""
    echo "─── SimDINOv2-B/16  Régime=full  Fraction=$FRAC  Seed=$SEED ───"

    python scripts/datacurve_one_run.py \
        --config "$CONFIG" \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --emb-dir "$OUT_DIR/embeddings" \
        --skip-if-done

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] run simdinov2_vitb16/full seed=$SEED a échoué (exit=$EXIT_CODE)" >&2
        echo "  → loggé, continuation des autres seeds." >&2
    fi
done

# ── Résumé f1_macro_pres_test des 3 seeds ────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "[résumé] SimDINOv2-B/16 — Régime=full — f1_macro_pres_test par seed"
echo "═══════════════════════════════════════════════"
PCT=$(python3 -c "print(int(round(${FRAC}*100)))")
python3 - "$OUT_DIR" "$PCT" <<'PY'
import json, os, sys
out_dir, pct = sys.argv[1], int(sys.argv[2])
vals = []
for seed in (0, 1, 2):
    mp = os.path.join(out_dir, "runs", f"frac{pct:03d}_seed{seed}", "metrics.json")
    if not os.path.exists(mp):
        print(f"  seed{seed}: (metrics.json absent — run incomplet)")
        continue
    m = json.load(open(mp))
    f1 = m.get("f1_macro_pres_test")
    f8 = m.get("f1_macro_8cls_test")
    vals.append(f1)
    print(f"  seed{seed}: f1_macro_pres_test={f1:.4f}  f1_macro_8cls_test={f8:.4f}  best_C={m.get('best_C')}")
if vals:
    import statistics as st
    mean = sum(vals) / len(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  → [simdinov2_vitb16/full] f1_macro_pres_test = {mean:.4f} ± {sd:.4f}  (n={len(vals)} seeds)")
else:
    print("  → [simdinov2_vitb16/full] aucun run complété.")
PY

echo ""
echo "[slurm] Job $SLURM_JOB_ID (simdinov2_vitb16/full) terminé."
echo "  Résultats : $OUT_DIR/runs/   |   Embeddings : $OUT_DIR/embeddings/"
SCRATCH_FREE=$(df -h "$SCRATCH" 2>/dev/null | awk 'NR==2{print $4}')
echo "[slurm] Espace scratch disponible : $SCRATCH_FREE"
