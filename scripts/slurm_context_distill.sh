#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Self-distillation contexte→tuile — DINOv3-B LoRA student, 3 seeds SÉQUENTIELS
# dans UN SEUL job (une seule allocation GPU, tuiles+contexte extraits une fois).
#
# ⚠️ MIG (a100_3g.20gb), PAS un A100 complet — GPU minimal jugé suffisant. Le
# teacher (DINOv3-L externe OU copie EMA du student) ne fait qu'un FORWARD, jamais
# de backward ni d'état optimizer — nettement moins coûteux qu'un entraînement. Le
# précédent le plus proche mesuré dans ce dépôt (slurm_lora_dinov3_vitl.sh) ENTRAÎNE
# (forward+backward+optimizer LoRA) un DINOv3-L COMPLET dans un slice a100_3g.20gb
# (20 Go) — mon student est plus petit (DINOv3-B) et le teacher n'ajoute qu'un
# forward. Jugement raisonné, PAS MESURÉ sur ce script précis : si OOM malgré tout,
# replier sur --gres=gpu:a100:1 + --mem=60G (A100 complet, 40 Go, queue plus lente —
# même repli que slurm_lora_dinov3_vitl.sh:34).
#
# ⚠️ --time NON MESURÉ. Bornes connues du dépôt : DINOv3-B LoRA SEUL (sans teacher)
# ≈1h20-1h30/seed (slurm_lora_rank_ablation.sh) ; DINOv3-L LoRA SEUL (entraîné, donc
# strictement plus cher qu'un simple forward de même taille) ≈5h56/seed
# (slurm_lora_dinov3_vitl.sh, job 66522732). Mon job = DINOv3-B entraîné + UN forward
# supplémentaire (DINOv3-L ou EMA-B) par step → borné par la 2e mesure par prudence :
# 3 seeds × ~6h ≈ 18h → --time=20:00:00 (marge). Design B ajoute un 3e forward
# (student sur le contexte, même petit modèle) — même budget par prudence, non mesuré.
# À AJUSTER après le premier run réel (logs/context_distill_*.out) — ne pas
# resoumettre à l'aveugle si le job time-out avant la fin des 3 seeds.
#
# ⚠️ --mem (RAM système, PAS VRAM) UNIFORME quels que soient contexte/design/teacher :
# context_crop.py REDIMENSIONNE tout contexte à out-size=224 AVANT sauvegarde (cf.
# docstring de context_crop.py) — le tenseur chargé en entraînement a la MÊME taille
# peu importe --context-size.
#
# Usage : $1 = context_size (512|1024|2048, défaut 1024 = R1). $2 = design (A|B,
# défaut A). $3 = teacher (nom d'un backbone frozen, défaut dinov3_vitl16_lvd ; ou
# 'ema_self'). $4 = config YAML (défaut configs/context_distill_dinov3b.yaml).
# PAS d'argument seed — les 3 seeds (0,1,2) tournent séquentiellement
# dans CE job, sur la même A100, avec les tuiles+contexte extraits UNE SEULE fois
# (convention scripts/slurm_lora_rank_ablation.sh, scripts/slurm_datacurve_spatial.sh).
#
# R1/R2/R3 ci-dessous = les 3 lignes du tableau "plan final" de l'utilisateur (PAS
# des tailles de contexte) — cf. scripts/context_distill_README.md §1 pour la
# convention de nommage complète. Aucun ordre imposé entre elles, jobs indépendants :
#   - R1 (1024px, DINOv3-L, Design A — l'expérience principale) :
#        sbatch scripts/slurm_context_distill.sh
#   - R2 (Design B — prérequis contexte val/test, cf. README §14) :
#        sbatch scripts/slurm_context_distill.sh 1024 B
#   - R3 (EMA self-teacher — aucun prérequis) :
#        sbatch scripts/slurm_context_distill.sh 1024 A ema_self
#   - Student SimDINOv2-B (iNat21 Plantae) + teacher SimL, Design B, contexte
#     512px (l'expérience demandée 2026-09 — cf. README §18 ; checkpoints pris
#     dans le config: context_distill_simdinov2b.yaml) :
#        sbatch scripts/slurm_context_distill.sh 512 B simdinov2_vitl16 \
#               configs/context_distill_simdinov2b.yaml
#   - Extension optionnelle, hors tableau (effet de la taille de contexte, comparée
#     à la baseline spatiale F1=0.4827±0.0042, frac100 LoRA r=8, mesuré 2026-08-29
#     depuis results/spatial_datacurve_CANONICAL.csv, cf. README §3) :
#        sbatch scripts/slurm_context_distill.sh 512
#        sbatch scripts/slurm_context_distill.sh 2048
#
# PRÉ-REQUIS :
#   - $SCRATCH/tiles.zip                      (tuiles 224px, convention existante)
#   - $SCRATCH/context_<size>.zip   (produit par scripts/context_crop.py EN LOCAL puis
#     transféré — cf. slurm_context_crop.sh). R2 (Design B) : ce zip doit AUSSI
#     contenir les crops val/test, pas seulement train (contrairement à R1/R3 et aux
#     variantes de contexte 512/2048px).
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitb16-pretrain-lvd1689m/
#   - $SCRATCH/hf_cache/models--facebook--dinov3-vitl16-pretrain-lvd1689m/ (sauf si
#     $3=ema_self, auquel cas aucun teacher externe n'est chargé)
#   - Student SimDINOv2 (configs/context_distill_simdinov2b.yaml) :
#     $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth (student) ET
#     $SCRATCH/checkpoints/simdinov2_vitl_inat21plantae.pth (teacher SimL) —
#     PAS de HF (le dépôt sslplant est déjà cloné par les runs frozen).
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=context_distill
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/context_distill_%j.out
#SBATCH --error=logs/context_distill_%j.err
#SBATCH --account=def-bouguess_gpu

CONTEXT_SIZE="${1:-1024}"
DESIGN="${2:-A}"
TEACHER="${3:-dinov3_vitl16_lvd}"
CONFIG="${4:-configs/context_distill_dinov3b.yaml}"
CODE_DIR="$HOME/benchmark-memoire"
VENV="$HOME/ENV/bin/activate"
OUT_DIR="$SCRATCH/context_distill"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "context_distill | context_size=$CONTEXT_SIZE design=$DESIGN teacher=$TEACHER | 3 seeds séquentiels"
echo "Nœud : $SLURMD_NODENAME | Job : $SLURM_JOB_ID"
echo "OUT_DIR : $OUT_DIR"
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

CONTEXT_ZIP="$SCRATCH/context_${CONTEXT_SIZE}.zip"
[[ ! -f "$CONTEXT_ZIP" ]] && {
    echo "[ERROR] $CONTEXT_ZIP introuvable — lancer context_crop.py EN LOCAL et transférer"
    echo "        les context_<size>.zip d'abord (cf. scripts/slurm_context_crop.sh)."
    exit 1
}
echo "[slurm] Extraction contexte ($CONTEXT_ZIP) → $SLURM_TMPDIR ..."
unzip -q "$CONTEXT_ZIP" -d "$SLURM_TMPDIR/"
# Design B (et toute taille où le contexte val/test est requis à l'inférence) :
# merger le zip val/test s'il existe — même convention que slurm_context_frozen_models.sh
# (le context_<size>.zip ne porte QUE le train ; val/test vit dans *_valtest.zip).
if [[ -n "$DESIGN" && "$DESIGN" == "B" && -f "$SCRATCH/context_${CONTEXT_SIZE}_valtest.zip" ]]; then
    echo "[slurm] Design B : merge val/test ($SCRATCH/context_${CONTEXT_SIZE}_valtest.zip) ..."
    unzip -q -o "$SCRATCH/context_${CONTEXT_SIZE}_valtest.zip" -d "$SLURM_TMPDIR/"
fi
CONTEXT_DIR="$SLURM_TMPDIR/context_${CONTEXT_SIZE}"
[[ ! -d "$CONTEXT_DIR" ]] && { echo "[ERROR] $CONTEXT_DIR absent après unzip"; exit 1; }
N_CROPS=$(find "$CONTEXT_DIR" -name "*.png" 2>/dev/null | wc -l)
echo "[slurm] context_${CONTEXT_SIZE} : $N_CROPS crops au total"

EMA_ARGS=()
if [[ "$TEACHER" == "ema_self" ]]; then
    EMA_ARGS=(--ema-momentum 0.999)
fi

# 3 seeds séquentiels sur la même allocation GPU — tuiles+contexte déjà extraits
# une seule fois ci-dessus (convention slurm_lora_rank_ablation.sh).
for SEED in 0 1 2; do
    SPATIAL_CSV_DIR="$CODE_DIR/spatial_datacurve/splits/frac100_seed${SEED}"
    [[ ! -f "$SPATIAL_CSV_DIR/train.csv" ]] && {
        echo "[ERROR] split spatial absent: $SPATIAL_CSV_DIR/train.csv — seed=$SEED sauté" >&2
        continue
    }

    # Design B : vérifie que le contexte val EXISTE (pas seulement train) — sinon
    # échec tardif et confus au milieu de l'entraînement.
    if [[ "$DESIGN" == "B" ]]; then
        FIRST_VAL_FP=$(sed -n '2p' "${SPATIAL_CSV_DIR}/val.csv" | cut -d',' -f1)
        if [[ -z "$FIRST_VAL_FP" || ! -f "$CONTEXT_DIR/$FIRST_VAL_FP" ]]; then
            echo "[ERROR] Design B nécessite le contexte pour val ET test, pas seulement train."
            echo "        Fichier attendu introuvable : $CONTEXT_DIR/$FIRST_VAL_FP"
            echo "        Relancer context_crop.py EN LOCAL sur val.csv/test.csv (--out-dir le même"
            echo "        dossier out/context) et retransférer context_${CONTEXT_SIZE}.zip."
            exit 1
        fi
    fi

    CFG_OVERRIDE="$SLURM_TMPDIR/cfg_context_distill_seed${SEED}_ctx${CONTEXT_SIZE}_d${DESIGN}.yaml"
    cat "$CONFIG" > "$CFG_OVERRIDE"
    cat >> "$CFG_OVERRIDE" <<EOF

# ── Override généré par slurm_context_distill.sh ───────────────────────────────
paths_narval:
  csv_dir: ${SPATIAL_CSV_DIR}
  tiles_dir: ${SLURM_TMPDIR}/tiles
EOF

    mkdir -p "$OUT_DIR"
    cp "$CFG_OVERRIDE" "$OUT_DIR/config_used_${SLURM_JOB_ID}_seed${SEED}_ctx${CONTEXT_SIZE}_d${DESIGN}.yaml"

    echo ""
    echo "─── seed=$SEED context_size=$CONTEXT_SIZE design=$DESIGN teacher=$TEACHER ───"

    python scripts/context_distill.py \
        --config "$CFG_OVERRIDE" \
        --teacher "$TEACHER" \
        --context-size "$CONTEXT_SIZE" \
        --context-dir "$CONTEXT_DIR" \
        --design "$DESIGN" \
        "${EMA_ARGS[@]}" \
        --seed "$SEED" \
        --out-dir "$OUT_DIR" \
        --skip-if-done

    if [[ $? -ne 0 ]]; then
        echo "[ERROR] seed=$SEED échoué — continuation" >&2
    fi
done

echo ""
echo "[slurm] context_size=$CONTEXT_SIZE design=$DESIGN teacher=$TEACHER terminé (3 seeds)."
echo "  Résultats sous : $OUT_DIR/runs/  (tag = {model}_ctxdistill_d${DESIGN}_t*_ctx${CONTEXT_SIZE}_r2a4_frac100_seed{0,1,2})"
