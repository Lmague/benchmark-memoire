#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Migration ponctuelle (une seule fois) — réaligne les run_dir LoRA existants sous
# $SCRATCH/sota_screening/lora_3models/runs/ vers le nouveau format de tag
# {model}_{regime}_frac{XXX}_seed{N} (= ckpt_tag), attendu par scripts/datacurve_one_run.py
# après le fix de collision de chemin (run_dir n'était scopé que par frac+seed avant,
# pas par modèle — cf. commit associé pour le détail du bug et la perte de données).
#
# Ne déplace RIEN sans vérifier d'abord que metrics.json existe et que son ckpt_tag
# correspond exactement au modèle attendu. Refuse d'écraser une destination déjà
# présente. Idempotent (relançable sans risque si déjà migré, ou partiellement migré).
#
# À EXÉCUTER MANUELLEMENT sur Narval (login node, pas besoin de sbatch) :
#   module load python/3.11   # si python3 n'est pas déjà dans le PATH du login node
#   bash scripts/migrate_lora3models_run_dirs.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

OUT_DIR="$SCRATCH/sota_screening/lora_3models"
RUNS="$OUT_DIR/runs"

migrate_one() {
    local src="$1" expected_tag="$2"
    local dst="$RUNS/$expected_tag"

    if [[ ! -d "$src" ]]; then
        echo "[SKIP] source absente : $src"
        return 0
    fi
    if [[ -d "$dst" ]]; then
        echo "[SKIP] destination déjà présente : $dst (migration déjà faite ?)"
        return 0
    fi
    if [[ ! -f "$src/metrics.json" ]]; then
        echo "[ERREUR] $src/metrics.json introuvable — pas migré." >&2
        return 1
    fi

    local actual_tag
    actual_tag=$(python3 -c "import json; print(json.load(open('$src/metrics.json'))['ckpt_tag'])")
    if [[ "$actual_tag" != "$expected_tag" ]]; then
        echo "[ERREUR] $src : ckpt_tag='$actual_tag' != attendu '$expected_tag' — pas migré." >&2
        return 1
    fi
    if [[ ! -f "$src/done" ]]; then
        echo "[WARN] $src/done absent (run non finalisé ?) — migration quand même." >&2
    fi

    echo "[OK] $src  ->  $dst  (ckpt_tag=$actual_tag confirmé)"
    mv "$src" "$dst"
}

echo "── SimDINOv2 ViT-B (3 seeds, déjà réorganisés manuellement) ──"
for SEED in 0 1 2; do
    migrate_one "$OUT_DIR/simdinov2_vitb16/runs/frac100_seed${SEED}" \
                "simdinov2_vitb16_lora_frac100_seed${SEED}"
done

echo ""
echo "── DINOv3 ViT-L seed=0 (résultat valide, non écrasé) ──"
migrate_one "$RUNS/frac100_seed0" "dinov3_vitl16_lvd_lora_frac100_seed0"

echo ""
echo "── Nettoyage de l'arborescence workaround (si vide) ──"
if [[ -d "$OUT_DIR/simdinov2_vitb16" ]]; then
    if find "$OUT_DIR/simdinov2_vitb16" -mindepth 1 -print -quit | grep -q .; then
        echo "[SKIP] $OUT_DIR/simdinov2_vitb16 non vide — laissé en place, à vérifier manuellement."
        find "$OUT_DIR/simdinov2_vitb16" -mindepth 1
    else
        rmdir "$OUT_DIR/simdinov2_vitb16/runs" 2>/dev/null || true
        rmdir "$OUT_DIR/simdinov2_vitb16" 2>/dev/null || true
        echo "[OK] arborescence vide supprimée."
    fi
fi

echo ""
echo "── État final de $RUNS ──"
ls -la "$RUNS"

echo ""
echo "── Rappel ──"
echo "SimDINOv2 ViT-L seed=0 est PERDU (écrasé par DINOv3-L avant le fix) — rien à"
echo "migrer pour ce run, il doit être recalculé :"
echo "  sbatch scripts/slurm_lora_simdinov2_vitl.sh"
echo "(les seeds 1 et 2, s'ils existent déjà sous le bon tag, seront skippés via --skip-if-done)."
