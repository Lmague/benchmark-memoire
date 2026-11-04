#!/usr/bin/env python3
"""Re-probe canonique mono-thread de la courbe de données SPATIALE (21 runs).

Pourquoi ce script existe
--------------------------
Les F1 de `lora_spatial_v2/results_spatial_summary.csv` viennent du PROBE INTERNE
de `scripts/datacurve_one_run.py` (`_run_probe_and_metrics`) : même fonction de fit
(`src.utils.make_canonical_lr`, donc même solveur/grille que le protocole canonique),
mais AUCUN forçage mono-thread BLAS avant l'import numpy/sklearn, ni dans le script
ni dans `spatial_datacurve/slurm_datacurve_spatial_v2.sh` (vérifié : aucune
occurrence de OMP_NUM_THREADS / OPENBLAS / MKL dans les deux fichiers).

Or AGENTS.md §4.8 est explicite : le nombre de threads BLAS change l'ORDRE DE
RÉDUCTION DES SOMMES et donc la solution vers laquelle lbfgs converge, même quand
le solveur converge normalement — mesuré à 0,0015 d'écart sur DINOv3-B MHSA seed0
(0,4821 à 1 thread vs 0,4836 à 3 threads). C'est le même écart d'ordre de grandeur
que plusieurs comparaisons discutées dans `datacurves.pdf` §3 (ex. le Δ spatiale vs
aléatoire à 75 %, ~0,02, ou les écarts inter-seed eux-mêmes). Les 21 runs spatiaux
n'ont donc PAS le même niveau de garantie de reproductibilité que l'ablation de rang
LoRA, qui a reçu un re-probe mono-thread dédié
(`scripts/rapport/lora_rank_ablation_probe.py` → `lora_rank_ablation_CANONICAL.json`).

Ce script fait la même chose pour les 21 runs spatiaux : re-fit LogisticRegression
mono-thread sur les embeddings déjà sur disque (aucun réentraînement du backbone),
même grille C, mêmes 3 métriques que l'agrégation existante (F1 complet, F1 8cls,
F1 train-pres) + comparaison au chiffre interne pour quantifier l'écart réel.

Écrit results/spatial_datacurve_CANONICAL.json (+ .csv) — NE remplace PAS
`lora_spatial_v2/results_spatial_summary.csv`, pour permettre la comparaison
avant/après. Si l'écart est négligeable (<0,002, cohérent avec AGENTS.md §4.8),
remplacer les chiffres de `datacurves.tex` / `analyse_encadrant.tex` par ceux-ci et
le documenter ; sinon, ce sont CES chiffres qui doivent devenir la source
canonique du mémoire.

    python3 scripts/rapport/probe_spatial_canonical.py
    python3 scripts/rapport/probe_spatial_canonical.py --only frac100 2   # 1 seed de test
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]   # grille canonique complète (AGENTS.md §3)
MAX_ITER = 2000
CGRID_SUBSAMPLE = 20000

# Mêmes 8 classes fiables que datacurve_one_run.py:LABELS_8CLS (11-class scheme,
# RHOL déjà exclue) — ALDE BIRC LICH MOSS PETF SEDG TUSS WILL.
LABELS_8CLS = [0, 2, 4, 5, 6, 8, 9, 10]
CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                   "RUBC", "SEDG", "TUSS", "WILL"]

FRAC_TAGS = ["frac001", "frac005", "frac010", "frac025", "frac050", "frac075", "frac100"]
SEEDS = [0, 1, 2]

EMB_ROOT = PROJ / "lora_spatial_v2"
SPLITS_ROOT = PROJ / "spatial_datacurve" / "splits"
MANIFEST_PATH = PROJ / "spatial_datacurve" / "manifest.json"


def _make_lr(c):
    try:
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                   random_state=SEED, multi_class="multinomial")
    except (TypeError, ValueError):
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)


def emb_dir_of(tag, s):
    # NB (AGENTS.md / aggregate_spatial_runs.py) : tous les dossiers de run internes
    # portent "frac100" (artefact --fraction 1.0) ; le niveau réel vient du dossier
    # PARENT (frac001…frac100), pas du nom du run.
    return EMB_ROOT / tag / "embeddings" / f"dinov3_vitb16_lvd_lora_r8a16_frac100_seed{s}"


def load_split_manifest(tag, s):
    p = SPLITS_ROOT / f"{tag}_seed{s}" / "manifest.json"
    with open(p) as f:
        return json.load(f)


def probe_one(tag, s):
    emb_dir = emb_dir_of(tag, s)
    data = {}
    for split in ("train", "val", "test"):
        e = np.load(emb_dir / f"{split}.npy").astype(np.float32)
        l = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64).ravel()
        data[split] = (e, l)

    sc = StandardScaler()
    xtr = sc.fit_transform(data["train"][0])
    xva = sc.transform(data["val"][0])
    xte = sc.transform(data["test"][0])
    ltr, lva, lte = data["train"][1], data["val"][1], data["test"][1]

    if xtr.shape[0] > CGRID_SUBSAMPLE:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(xtr.shape[0], CGRID_SUBSAMPLE, replace=False)
        xtr_cg, ltr_cg = xtr[idx], ltr[idx]
    else:
        xtr_cg, ltr_cg = xtr, ltr

    best_c, best_f1v = None, -1.0
    for c in C_GRID:
        clf = _make_lr(c); clf.fit(xtr_cg, ltr_cg)
        f1v = f1_score(lva, clf.predict(xva), average="macro", zero_division=0,
                       labels=list(range(11)))
        if f1v > best_f1v:
            best_f1v, best_c = f1v, c

    clf_final = _make_lr(best_c); clf_final.fit(xtr, ltr)
    pred_te = clf_final.predict(xte)

    present_te = sorted({int(v) for v in lte})
    f1_pres = float(f1_score(lte, pred_te, average="macro", zero_division=0,
                             labels=present_te))
    f1_8cls = float(f1_score(lte, pred_te, average="macro", zero_division=0,
                             labels=LABELS_8CLS))
    f1_per_class = {CLASS_NAMES_11[i]: float(v) for i, v in enumerate(
        f1_score(lte, pred_te, average=None, zero_division=0, labels=list(range(11))))}

    split_manifest = load_split_manifest(tag, s)
    present_train = [c for c, n in split_manifest.get("classes_11cls", {}).items()
                     if n > 0 and c in f1_per_class]
    f1_train_pres = float(np.mean([f1_per_class[c] for c in present_train])) \
        if present_train else float("nan")

    return {
        "f1_macro_pres_test": f1_pres, "f1_macro_8cls_test": f1_8cls,
        "f1_macro_train_pres": round(f1_train_pres, 4), "best_C": best_c,
        "n_train": int(xtr.shape[0]), "n_classes_train": len(present_train),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=2, metavar=("TAG", "SEED"), default=None,
                    help="probe un seul (tag, seed), ex. --only frac100 2")
    args = ap.parse_args()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    level_info = {}
    for nv in manifest["niveaux"]:
        for s in nv["seeds"]:
            level_info[(nv["tag"], s["seed"])] = {
                "cible": nv["cible"], "fraction_reelle": s["fraction_reelle"]}

    if args.only:
        tag, s = args.only[0], int(args.only[1])
        out = probe_one(tag, s)
        old = None
        try:
            with open(EMB_ROOT / "results_spatial_summary.csv") as f:
                for row in csv.DictReader(f):
                    if row["seed"] == str(s) and abs(
                            float(row["fraction_cible"]) - level_info[(tag, s)]["cible"]) < 1e-9:
                        old = float(row["f1_macro_pres_test"])
        except FileNotFoundError:
            pass
        print(f"[{tag} seed={s}] F1={out['f1_macro_pres_test']:.4f} C={out['best_C']} "
              f"(interne={old})")
        return

    t0 = time.time()
    rows = []
    for tag in FRAC_TAGS:
        for s in SEEDS:
            key = (tag, s)
            if key not in level_info:
                print(f"[SKIP] {tag} seed{s} absent de manifest.json")
                continue
            out = probe_one(tag, s)
            rows.append({
                "fraction_cible": level_info[key]["cible"],
                "spatial_fraction_reelle": level_info[key]["fraction_reelle"],
                "seed": s, "tag": tag, **out,
            })
            print(f"[{tag} seed={s}] F1={out['f1_macro_pres_test']:.4f} "
                  f"8cls={out['f1_macro_8cls_test']:.4f} "
                  f"train_pres={out['f1_macro_train_pres']:.4f} C={out['best_C']}")

    rows.sort(key=lambda r: (r["fraction_cible"], r["seed"]))
    out_json = PROJ / "results" / "spatial_datacurve_CANONICAL.json"
    with open(out_json, "w") as f:
        json.dump({"protocol": "LR lbfgs multinomial, C_grid [1e-4..10], seed=42, "
                               "max_iter=2000, BLAS mono-thread (AGENTS.md §4.8) — "
                               "re-probe des embeddings de lora_spatial_v2/ (aucun "
                               "réentraînement)", "date": time.strftime("%Y-%m-%d"),
                   "runs": rows}, f, indent=2)
    out_csv = PROJ / "results" / "spatial_datacurve_CANONICAL.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out_json}\n-> {out_csv}")

    # Comparaison au probe interne existant, si présent
    old_csv = EMB_ROOT / "results_spatial_summary.csv"
    if old_csv.exists():
        old = {(r["seed"], round(float(r["fraction_cible"]), 4)): float(r["f1_macro_pres_test"])
               for r in csv.DictReader(open(old_csv))}
        print(f"\n{'='*64}\nÉcart re-probe canonique vs probe interne (f1_macro_pres_test)\n{'='*64}")
        max_gap = 0.0
        for r in rows:
            k = (str(r["seed"]), round(r["fraction_cible"], 4))
            if k in old:
                gap = r["f1_macro_pres_test"] - old[k]
                max_gap = max(max_gap, abs(gap))
                print(f"  {r['tag']} seed{r['seed']}: canonique={r['f1_macro_pres_test']:.4f} "
                      f"interne={old[k]:.4f} Δ={gap:+.4f}")
        print(f"\nÉcart max observé : {max_gap:.4f} "
              f"({'cohérent avec' if max_gap < 0.003 else 'AU-DESSUS de'} "
              f"AGENTS.md §4.8, ~0,0015 attendu)")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
