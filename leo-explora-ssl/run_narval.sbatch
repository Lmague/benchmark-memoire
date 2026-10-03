#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ExPLoRA SSL — SimDINOv2 ViT-B/16 (iNat-Plantae) → tuiles Léo (4 orthos, ~355k).
#
# Extended pre-training NON SUPERVISÉ (DINO + iBOT + KoLeo) pour adapter le
# backbone au domaine aérien nadir. Ce n'est PAS le régime « explora_like »
# SUPERVISÉ exclu des rapports (scripts/rapport/registry.py : EXCLUDED_MODELS) —
# ici c'est le vrai protocole ExPLoRA (Khanna et al. 2024, arXiv:2406.10973).
#
# Recette fidèle au papier : blocs 0..9 LoRA Q,V (r=16) + 2 derniers blocs
# full-FT + LayerNorms dégelées ; teacher EMA. Multi-crop = 2 GLOBALES distinctes
# (le bug « 1 seule globale » est corrigé, cf. ssl_train.py) + 6 locales.
#
# 3 SEEDS (array 0-2) : requis pour l'ensemble décisionnel de l'outil
# d'annotation (moyenne des probas par espèce). Sorties suffixées _seed{N},
# convention du dépôt Mémoire.
#
# PRÉ-REQUIS sur Narval (déjà présents pour les embeddings gelés) :
#   $SCRATCH/web/tiles/{clairiere,maison,trail,rousseau}/   (~355 486 tuiles)
#   $SCRATCH/web/manifest_{clairiere,maison,trail,rousseau}.json
#   $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth
# (pas d'export brut : on réutilise web/tiles, le jeu canonique des embeddings gelés)
# $HOME/benchmark-memoire/leo-explora-ssl = ce dépôt (git pull à jour).
# $HOME/ENV = env standard du dépôt Mémoire (torch + einops), utilisé par défaut.
#
# Durée : ~27 h/seed sur A100 entier ; ~60-65 h/seed sur MIG 3g.20gb (observé
# ~0,41 s/step à batch 32). --time=72h pour couvrir la queue du cosine.
# Mémoire : 2 globales + 6 locales ; batch 64 par défaut sur MIG (128 si GPU plein).
# Pour un A100 entier : passer --gres=gpu:a100:1 et BATCH_OVERRIDE=128.
# Soumission : sbatch run_narval.sbatch
# ═══════════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=leo_explora_ssl
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/leo_ssl_%A_%a.out
#SBATCH --error=logs/leo_ssl_%A_%a.err
#SBATCH --account=def-bouguess_gpu

SEED=${SEED:-$SLURM_ARRAY_TASK_ID}
REPO="$HOME/benchmark-memoire/leo-explora-ssl"
DATA="$SCRATCH/leo_ssl"
VENV="${VENV:-$HOME/ENV/bin/activate}"
BATCH="${BATCH_OVERRIDE:-64}"

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SCRATCH/torch_cache"

echo "═══════════════════════════════════════════════"
echo "ExPLoRA SSL — SimDINOv2-B → tuiles Léo | seed=$SEED (task ${SLURM_ARRAY_TASK_ID:-?})"
echo "Job : ${SLURM_JOB_ID:-local} | Noeud : ${SLURMD_NODENAME:-?} | batch=$BATCH"
echo "═══════════════════════════════════════════════"

[[ ! -d "$REPO" ]] && { echo "[ERROR] REPO absent: $REPO"; exit 1; }
[[ -f "$VENV" ]] || { echo "[ERROR] venv absent: $VENV (export VENV=...)"; exit 1; }
WEB="$SCRATCH/web"
CKPT="$SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth"
for s in clairiere maison trail rousseau; do
  [[ -f "$WEB/manifest_$s.json" ]] || { echo "[ERROR] MANQUANT: $WEB/manifest_$s.json"; exit 1; }
done
[[ -d "$WEB/tiles" ]] || { echo "[ERROR] MANQUANT: $WEB/tiles/"; exit 1; }
[[ -f "$CKPT" ]] || { echo "[ERROR] MANQUANT: $CKPT  (rsync ~/Documents/Mémoire/checkpoints/simdinov2_vitb_inat21plantae.pth)"; exit 1; }
echo "[check] tuiles web : $(find "$WEB/tiles" -name '*.jpg' | wc -l) (attendu ~355486)"

module load python/3.11 cuda/12.2 cudnn/8.9
source "$VENV"
cd "$REPO"
mkdir -p "$DATA/runs" logs

# Résout $SCRATCH dans la config + force batch (la config garde epochs=50)
sed "s|\$SCRATCH|$SCRATCH|g" configs/leo_vitb16_ssl.json > "$DATA/config_resolved_seed${SEED}.json"
python - "$DATA/config_resolved_seed${SEED}.json" "$BATCH" "$SEED" <<'PYEOF'
import json, sys
p, b, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = json.loads(open(p).read()); c["batch"] = b; c["seed"] = s
json.dump(c, open(p, "w"), indent=2)
print(f"[config] {p} batch={b} seed={s}")
PYEOF

# Préflight : construit le backbone + applique ExPLoRA (échec rapide < 5 min)
python - "$DATA/config_resolved_seed${SEED}.json" <<'PYEOF' || { echo "[ERROR] préflight échoué — job stoppé"; exit 1; }
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from ssl_train import load_backbone, apply_explora
cfg = json.loads(Path(sys.argv[1]).read_text())
bb = load_backbone(Path(cfg["simdinov2_ckpt"]))
g = apply_explora(bb, cfg.get("lora_r", 16), cfg.get("lora_alpha", 16.0),
                  cfg.get("n_full_blocks", 2))
n = sum(p.numel() for ps in g.values() for p in ps)
tot = sum(p.numel() for p in bb.parameters())
print(f"[préflight] OK LoRA+late+norms = {n:,} / {tot:,} ({100*n/tot:.2f}%)")
PYEOF

OUT_SEED="$DATA/runs/leo_vitb16_ssl_seed${SEED}"
echo "[run] seed=$SEED batch=$BATCH → $OUT_SEED/checkpoints/{ep*,last}.pth"
python ssl_train.py --config "$DATA/config_resolved_seed${SEED}.json" --seed "$SEED"
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "[WARN] seed=$SEED batch=$BATCH échoué (exit=$RC), retry batch /2" >&2
  B2=$((BATCH / 2))
  python - "$DATA/config_resolved_seed${SEED}.json" "$B2" <<'PYEOF'
import json, sys
p, b = sys.argv[1], int(sys.argv[2])
c = json.loads(open(p).read()); c["batch"] = b
json.dump(c, open(p, "w"), indent=2)
PYEOF
  python ssl_train.py --config "$DATA/config_resolved_seed${SEED}.json" --seed "$SEED"
  RC=$?
  [[ $RC -ne 0 ]] && { echo "[ERROR] retry échoué aussi"; exit $RC; }
fi

echo ""
echo "[slurm] seed=$SEED terminé."
echo "  checkpoints : ls -lh $OUT_SEED/checkpoints/"
echo "  Rapatriement : rsync -avP narval:$OUT_SEED/checkpoints/ ~/Documents/Mémoire/results/leo_explora_ssl/seed${SEED}/"
echo "[slurm] done."
