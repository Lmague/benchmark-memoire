#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Texture — extraction (7 familles) + ablation à sonde canonique — CPU-only, Narval.
#
# Enchaîne, dans un seul job :
#   1. extraction des features de texture des 3 splits  (scripts/texture_features.py)
#   2. contrôle de non-régression : la baseline doit reproduire le F1 du registre
#   3. ablation famille par famille + combinée, bootstrap apparié
#      (scripts/texture_ablation.py)
#
# AUCUN GPU : numpy/scipy/pywt pour l'extraction, sklearn lbfgs CPU pour la sonde.
# Le job est REPARTABLE : l'extraction saute les chunks déjà écrits et l'ablation saute
# les modèles dont le JSON de sortie existe. Un job tué par le time limit se relance tel
# quel (mais voir la note « temps » plus bas : mieux vaut viser juste du premier coup).
#
# AVANT SOUMISSION — éditer la section « Configuration », puis :
#   mkdir -p logs && sbatch scripts/slurm_texture.sh
#   (mkdir logs n'est nécessaire qu'une fois ; le dépôt ne versionne pas logs/.)
#
# Pré-requis présents sous $SCRATCH (mêmes conventions que les autres campagnes) :
#   $SCRATCH/tiles.zip          — tuiles 224px (dézippées dans $SLURM_TMPDIR)
#   $SCRATCH/splits/            — {train,val,test}_11cls.csv — ORDRE DES EMBEDDINGS
#   $SCRATCH/embeddings/        — <modèle>_{train,val,test}{,_labels}.npy
#
# Sortie : $SCRATCH/texture/{train,val,test}.{npy,json} + ablation_<tag>.json
#          → rapatrier en local dans results/texture/ (commande en fin de job).
#
# ── ⚠️ TEMPS : c'est la SONDE qui coûte, pas l'extraction ─────────────────────
# Mesuré en local (1 thread BLAS, obligatoire — AGENTS.md §4.8) : ablation COMPLÈTE d'un
# modèle = baseline + 7 familles + combinée = 9 jeux × (7 valeurs de C pour la sonde 11cls
# + 7 pour la sonde 8cls séparée) = 126 ajustements lbfgs ≈ 47 min. L'extraction des
# 80 088 tuiles prend ~5 min à 16 processus. Compter ~1 h par entrée de MODELS_SPEC.
# 8 h de limite couvrent donc ~8 modèles.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=texture
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=logs/texture_%j.out
#SBATCH --error=logs/texture_%j.err
#SBATCH --account=def-bouguess   # ← éditer : votre compte Alliance/Narval
# PAS de --gres=gpu : 100 % CPU. Le parallélisme est par PROCESSUS (extraction) et par
# cœurs pour scipy/sklearn ; BLAS reste mono-thread pour les sondes (chiffres canoniques).

# ─── Configuration — À ÉDITER (ou surcharger par variable d'environnement) ────
# Les valeurs par défaut correspondent à Narval ; toute variable déjà définie dans
# l'environnement est conservée, ce qui permet un dry-run local :
#   CODE_DIR=$PWD VENV=/tmp/stub.sh SPLITS_DIR=$PWD/splits TILES_ZIP=/tmp/tiles.zip \
#     OUT_DIR=/tmp/tex_out SKIP_ABLATION=1 LIMIT="--limit 300" bash scripts/slurm_texture.sh
CODE_DIR="${CODE_DIR:-$HOME/benchmark-memoire}"
VENV="${VENV:-$HOME/ENV/bin/activate}"
SPLITS_DIR="${SPLITS_DIR:-$SCRATCH/splits}"
TILES_ZIP="${TILES_ZIP:-$SCRATCH/tiles.zip}"
EMB_DIR="${EMB_DIR:-$SCRATCH/embeddings}"
OUT_DIR="${OUT_DIR:-$SCRATCH/texture}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"   # processus pour l'extraction
N_BOOT="${N_BOOT:-1000}"                        # tirages du bootstrap apparié
SKIP_EXTRACTION="${SKIP_EXTRACTION:-0}"          # 1 = réutiliser un cache déjà extrait
SKIP_CONTROL="${SKIP_CONTROL:-0}"                # 1 = sauter le contrôle de non-régression
SKIP_ABLATION="${SKIP_ABLATION:-0}"              # 1 = extraction seule (validation rapide)
SPLITS_TO_DO="${SPLITS:-train val test}"          # sous-ensemble de splits à extraire
LIMIT="${LIMIT:-}"                               # ex. LIMIT="--limit 300" pour un dry-run
# TILES_DIR : laissé vide ici. Il est résolu dans la section extraction, à partir de
# $SLURM_TMPDIR — un run « ablations seules » (SKIP_EXTRACTION=1) n'a pas besoin des
# tuiles, donc le résoudre ici ferait échouer un dry-run local pour rien.
TILES_DIR="${TILES_DIR:-}"

# Modèles à ablater, format "<préfixe des .npy>:<schéma de labels>", UN PAR LIGNE.
#   - gelés  : préfixe dans $EMB_DIR, labels 12 classes (RHOL absente) → 12cls
#   - affinés: préfixe d'un dossier de run, labels déjà 11 classes  → 11cls
# Chaque entrée coûte ~1 h. Par défaut : les deux gelés de référence (SimB = celui qui
# « marchait très bien en frozen » ; DINOv3-B = le backbone de référence du dépôt).
# Surchargeable : MODELS_SPEC="$EMB_DIR/xxx:12cls" bash scripts/slurm_texture.sh ...
if [[ -z "${MODELS_SPEC:-}" ]]; then
    MODELS_SPEC="$EMB_DIR/simdinov2_vitb16:12cls
$EMB_DIR/dinov3_vitb16_lvd:12cls"
fi
# Exemples supplémentaires (décommenter et ajuster les chemins) :
#   $SCRATCH/ft_ssl_results/<...>_runs/<...>_frac100_seed0:11cls
#   $SCRATCH/sota_screening/full/embeddings/vitb16_full_frac100_seed0:11cls

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# ──────────────────────────────────────────────────────────────────────────────

set -u
echo "═══════════════════════════════════════════════"
echo "texture  | job ${SLURM_JOB_ID:-local} sur ${SLURMD_NODENAME:-localhost}"
echo "début   : $(date -Is)"
echo "═══════════════════════════════════════════════"

# ── Vérifications préliminaires ───────────────────────────────────────────────
fail() { echo "[ERROR] $*" >&2; exit 1; }
[[ -d "$CODE_DIR" ]] || fail "CODE_DIR introuvable : $CODE_DIR"
[[ -f "$VENV"     ]] || fail "venv introuvable : $VENV"
# Les tuiles et les CSV de split ne sont requis QUE si l'extraction va réellement tourner :
# un run « ablations seules » sur un cache déjà extrait ne doit pas exiger tiles.zip.
NEED_TILES=1
if [[ "$SKIP_EXTRACTION" == "1" && -f "$OUT_DIR/train.npy" ]]; then
    NEED_TILES=0
fi
if [[ "$NEED_TILES" == "1" ]]; then
    [[ -f "$TILES_ZIP" ]] || fail "$TILES_ZIP introuvable (tuiles 224px)"
    [[ -d "$SPLITS_DIR" ]] || fail "$SPLITS_DIR introuvable (CSV de split = ordre des embeddings)"
    for s in $SPLITS_TO_DO; do
        [[ -f "$SPLITS_DIR/${s}_11cls.csv" ]] || fail "CSV manquant : $SPLITS_DIR/${s}_11cls.csv"
    done
else
    echo "[slurm] extraction sautée → ni tiles.zip ni CSV de split requis"
fi
if [[ "$SKIP_ABLATION" != "1" ]]; then
    for s in $SPLITS_TO_DO; do
        [[ -f "$OUT_DIR/${s}.npy" ]] && continue
        [[ "$NEED_TILES" == "1" ]] && continue   # sera extrait juste après
        echo "[slurm] ATTENTION : cache $OUT_DIR/${s}.npy absent — l'ablation échouera" >&2
    done
fi

# ── Environnement Python ──────────────────────────────────────────────────────
module load python/3.11 || echo "[slurm] 'module' indisponible — ignoré (dry-run local ?)"
source "$VENV"
cd "$CODE_DIR"
mkdir -p logs "$OUT_DIR"

# Dépendances de src/texture.py : numpy (cœur), scipy (GLSZM/NGTDM), PyWavelets (DWT).
python - <<'PY' || fail "dépendances manquantes — lancer : pip install -r requirements.txt"
import importlib, sys
manquants = [m for m in ("numpy", "scipy", "pywt") if importlib.util.find_spec(m) is None]
if manquants:
    print("modules absents :", ", ".join(manquants), file=sys.stderr)
    sys.exit(1)
print("[slurm] dépendances OK (numpy, scipy, pywt)")
PY

# ── Tuiles → $SLURM_TMPDIR (SSD local) ────────────────────────────────────────
if [[ "$SKIP_EXTRACTION" != "1" || ! -f "$OUT_DIR/train.npy" ]]; then
    # Résolution de TILES_DIR ici seulement : hors SLURM, $SLURM_TMPDIR est vide et le
    # chemin deviendrait « /tiles » (unzip à la racine).
    if [[ -z "$TILES_DIR" ]]; then
        [[ -n "${SLURM_TMPDIR:-}" ]] || fail "SLURM_TMPDIR vide et TILES_DIR non fourni — hors SLURM : TILES_DIR=/chemin/vers/tiles"
        TILES_DIR="$SLURM_TMPDIR/tiles"
    fi
    echo ""
    echo "[slurm] Dézippage des tuiles → $(dirname "$TILES_DIR") ..."
    if [[ ! -d "$TILES_DIR" ]]; then
        unzip -q "$TILES_ZIP" -d "$(dirname "$TILES_DIR")" || fail "unzip a échoué"
    fi
    N_TILES=$(find "$TILES_DIR" -name '*.png' 2>/dev/null | wc -l)
    echo "[slurm] $N_TILES tuiles présentes dans $TILES_DIR"
    [[ "$N_TILES" -gt 0 ]] || fail "aucune tuile dans $TILES_DIR"
fi

# ── 1. Extraction des features de texture, par split ──────────────────────────
# Le CSV passé est celui de $SCRATCH/splits : c'est l'ORDRE DES EMBEDDINGS. Utiliser
# splits/ du dépôt produirait un cache désaligné (vérifié : l'ordre local == l'ordre
# des .npy, mais autant lever l'ambiguïté côté cluster).
EXTRACT_EXIT=0
if [[ "$SKIP_EXTRACTION" == "1" ]]; then
    echo ""
    echo "[slurm] Extraction sautée (SKIP_EXTRACTION=1) — cache existant réutilisé"
else
    for SPLIT in $SPLITS_TO_DO; do
        echo ""
        echo "─── Extraction split=${SPLIT} (n_jobs=${N_JOBS}) ───"
        # shellcheck disable=SC2086
        python scripts/texture_features.py \
            --split "$SPLIT" \
            --csv "$SPLITS_DIR/${SPLIT}_11cls.csv" \
            --tiles-dir "$TILES_DIR" \
            --out "$OUT_DIR/${SPLIT}.npy" \
            --n-jobs "$N_JOBS" \
            $LIMIT \
            || { echo "[ERROR] extraction ${SPLIT} échouée" >&2; EXTRACT_EXIT=1; break; }
    done
fi
[[ $EXTRACT_EXIT -eq 0 ]] || exit 1

# shellcheck disable=SC2086
python - "$OUT_DIR" $SPLITS_TO_DO <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
for s in sys.argv[2:]:
    meta = json.loads((out / f"{s}.json").read_text())
    bad = meta.get("n_tuiles_illisibles", 0)
    flag = f"  ⚠️ {bad} illisible(s)" if bad else ""
    print(f"[slurm] cache {s}: {meta['shape']}  non-finis={meta['n_non_finis']}  "
          f"csv={pathlib.Path(meta['csv']).name}{flag}")
PY

# ── 2. Contrôle de non-régression ─────────────────────────────────────────────
# La baseline embedding-seul DOIT reproduire le F1 canonique du registre (sinon le cache
# est désaligné ou le schéma de labels est faux, et toute l'ablation est à jeter).
# Valeurs de contrôle (registre all_models_canonical_merged.json, tolérance 0.002) :
#   simdinov2_vitb16     11cls 0.4723   8cls sonde séparée 0.6537
#   dinov3_vitb16_lvd    11cls 0.4712   8cls sonde séparée 0.6542
CONTROL_EXIT=0
if [[ "$SKIP_CONTROL" == "1" || "$SKIP_ABLATION" == "1" ]]; then
    echo ""
    echo "[slurm] Contrôle baseline sauté (SKIP_CONTROL=$SKIP_CONTROL, SKIP_ABLATION=$SKIP_ABLATION)"
else
    echo ""
    echo "─── Contrôle : baseline embedding seul ───"
    while IFS= read -r ENTRY; do
        [[ -z "$ENTRY" ]] && continue
        PREFIX="${ENTRY%:*}"; SCHEMA="${ENTRY##*:}"
        TAG="$(basename "$PREFIX")_controle"
        [[ -f "$OUT_DIR/ablation_${TAG}.json" ]] && { echo "  ${TAG} déjà fait → sauté"; continue; }
        python scripts/texture_ablation.py \
            --emb "$PREFIX" --label-schema "$SCHEMA" \
            --texture-dir "$OUT_DIR" --out-dir "$OUT_DIR" \
            --tag "$TAG" --baseline-only \
            || { echo "[ERROR] contrôle baseline échoué pour $PREFIX" >&2; CONTROL_EXIT=1; }
    done <<< "$MODELS_SPEC"
    python - "$OUT_DIR" <<'PY'
import json, sys, pathlib
ATTENDU = {"simdinov2_vitb16": (0.4723, 0.6537), "dinov3_vitb16_lvd": (0.4712, 0.6542)}
TOL = 0.002
out = pathlib.Path(sys.argv[1])
for f in sorted(out.glob("*_controle.json")):
    r = json.loads(f.read_text())["runs"]["baseline"]
    cle = next((k for k in ATTENDU if k in f.name), None)
    f11, f8 = r["f1_macro_pres"], r["f1_macro_8cls_sep"]
    if cle is None:
        print(f"  {f.name}: 11cls={f11:.4f} 8cls_sep={f8} (pas de valeur de contrôle)")
        continue
    a11, a8 = ATTENDU[cle]
    ok11, ok8 = abs(f11 - a11) <= TOL, (f8 is not None and abs(f8 - a8) <= TOL)
    print(f"  {cle:22s} 11cls={f11:.4f} (attendu {a11}, {'OK' if ok11 else 'ÉCART'})  "
          f"8cls_sep={f8 if f8 is None else round(f8, 4)} (attendu {a8}, {'OK' if ok8 else 'ÉCART'})")
    if not (ok11 and ok8):
        print(f"  ⚠️  ÉCART > {TOL} : cache désaligné ou schéma de labels erroné — "
              f"l'ablation serait à jeter.")
PY
fi

# ── 3. Ablation famille par famille + combinée ────────────────────────────────
if [[ "$SKIP_ABLATION" == "1" ]]; then
    echo ""
    echo "[slurm] Ablation sautée (SKIP_ABLATION=1) — mode validation d'extraction"
    echo "═══════════════════════════════════════════════"
    echo "[slurm] fin : $(date -Is)  (extraction=$EXTRACT_EXIT)"
    echo "═══════════════════════════════════════════════"
    exit $EXTRACT_EXIT
fi
echo ""
N_MODELS=$(grep -c . <<< "$MODELS_SPEC")
echo "─── Ablation (${N_MODELS} modèle(s), ${N_BOOT} tirages) ───"
ABL_EXIT=0
while IFS= read -r ENTRY; do
    [[ -z "$ENTRY" ]] && continue
    PREFIX="${ENTRY%:*}"; SCHEMA="${ENTRY##*:}"
    TAG="$(basename "$PREFIX")"
    OUT_JSON="$OUT_DIR/ablation_${TAG}.json"
    if [[ -f "$OUT_JSON" ]]; then
        echo "  ${TAG} : $OUT_JSON existe déjà → sauté (supprimer le fichier pour refaire)"
        continue
    fi
    echo ""
    echo "── ${TAG} (${SCHEMA}) ──"
    python scripts/texture_ablation.py \
        --emb "$PREFIX" --label-schema "$SCHEMA" \
        --texture-dir "$OUT_DIR" --out-dir "$OUT_DIR" \
        --tag "$TAG" --include-combined --n-boot "$N_BOOT" \
        || { echo "[ERROR] ablation échouée pour $PREFIX (les autres modèles continuent)" >&2
             ABL_EXIT=1; }
done <<< "$MODELS_SPEC"

# ── Récapitulatif ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "[slurm] Sorties dans $OUT_DIR :"
ls -la "$OUT_DIR" | tail -n +2
echo ""
echo "[slurm] Rapatriement en local (depuis VOTRE machine) :"
echo "  rsync -avP ${USER:-<user>}@narval.alliancecan.ca:$OUT_DIR/ results/texture/"
echo ""
echo "[slurm] fin : $(date -Is)  (extraction=$EXTRACT_EXIT contrôle=$CONTROL_EXIT ablation=$ABL_EXIT)"
echo "═══════════════════════════════════════════════"
[[ $ABL_EXIT -eq 0 ]] || exit 1
exit 0
