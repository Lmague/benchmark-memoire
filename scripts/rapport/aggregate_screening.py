#!/usr/bin/env python3
"""Phase 0.a — Agrégation de tous les runs fine-tunés locaux.

Lit les `metrics.json` des 11 familles de runs déclarées dans `registry.RUN_FAMILIES`
(84 runs sota_screening + 24 runs LoRA + 21 runs frac100 divers) et produit :

  results/rapport_data/screening_raw.csv        — une ligne par run
  results/rapport_data/screening_agg.csv        — modèle × fraction (moy/σ/min/max)
  results/rapport_data/per_class_by_regime.csv  — modèle × fraction × 11 classes
  results/rapport_data/per_class_all_models.csv — F1 par classe, tous modèles à 100 %
  results/rapport_data/frozen_probe_curve.csv   — courbe de probing gelé (copie annotée)

Attention (AGENTS.md §4.2) : le F1 des `metrics.json` est celui du **probe interne du
run** (grille C du pipeline d'entraînement). Il diffère légèrement du **probe canonique**
de `results/all_models_canonical_merged.json` (ex. LoRA 100 % : 0,4829 interne vs
0,4835 canonique). La colonne `probe` indique lequel est utilisé.

    python3 scripts/rapport/aggregate_screening.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from registry import (CLASSES_11, FROZEN_MODELS, OUT, RUN_FAMILIES, ensure_out, p,
                       runs_prefix)


def _stats(v):
    a = np.asarray(v, dtype=float)
    return dict(mean=float(a.mean()),
                std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                min=float(a.min()), max=float(a.max()), n=len(a))


def load_runs() -> list[dict]:
    """Toutes les lignes de run, tous pipelines confondus."""
    rows = []
    for key, fam in RUN_FAMILIES.items():
        display, runs_dir, _emb, mtype, init, pipeline = fam[:6]
        if runs_dir is None:
            # Familles sans runs locaux agrégeables (contexte R1/R2/R3 : leurs
            # metrics.json n'ont pas les champs fraction/best_epoch du pipeline
            # datacurve ; couverture via per_class_tier/cache + metrics.json direct).
            print(f"  [SKIP] {key}: pas de runs agrégeables (runs_dir=None)")
            continue
        pattern = p(runs_dir, f"{runs_prefix(key)}*", "metrics.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  [SKIP] {key}: aucun run sous {runs_dir}")
            continue
        for f in files:
            d = json.load(open(f))
            row = {
                "model": key, "display": display, "type": mtype, "init": init,
                "pipeline": pipeline,
                "fraction": float(d["fraction"]),
                "n_train_tiles": int(d["n_train_tiles"]),
                "seed": int(d["seed"]),
                "f1_pres": float(d["f1_macro_pres_test"]),
                "f1_pres_val": float(d.get("f1_macro_pres_val", float("nan"))),
                "f1_8cls": float(d.get("f1_macro_8cls_test", float("nan"))),
                "accuracy": float(d.get("accuracy_test", float("nan"))),
                "best_epoch": int(d.get("best_epoch", -1)),
                "best_C": float(d.get("best_C", float("nan"))),
                "probe": "interne_run",
                "run_dir": os.path.relpath(os.path.dirname(f), p()),
            }
            for c in CLASSES_11:
                row[f"f1_{c}"] = float(d.get("f1_per_class_test", {}).get(c, float("nan")))
            rows.append(row)
        print(f"  {key:28s} {len(files):3d} runs  "
              f"({len({r['fraction'] for r in rows if r['model'] == key})} fractions)")
    return rows


def write_raw(rows):
    cols = ["model", "display", "type", "init", "pipeline", "fraction", "n_train_tiles",
            "seed", "f1_pres", "f1_pres_val", "f1_8cls", "accuracy", "best_epoch",
            "best_C", "probe", "run_dir"] + [f"f1_{c}" for c in CLASSES_11]
    with open(os.path.join(OUT, "screening_raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["fraction"], r["seed"])):
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] screening_raw.csv ({len(rows)} runs)")


def write_agg(rows):
    keys = sorted({(r["model"], r["fraction"]) for r in rows})
    out = []
    for model, frac in keys:
        sub = [r for r in rows if r["model"] == model and r["fraction"] == frac]
        s = _stats([r["f1_pres"] for r in sub])
        rec = {
            "model": model, "display": sub[0]["display"], "type": sub[0]["type"],
            "init": sub[0]["init"], "pipeline": sub[0]["pipeline"],
            "fraction": frac, "n_train_tiles": sub[0]["n_train_tiles"], "n_seeds": s["n"],
            "f1_pres_mean": s["mean"], "f1_pres_std": s["std"],
            "f1_pres_min": s["min"], "f1_pres_max": s["max"],
            "f1_8cls_mean": float(np.mean([r["f1_8cls"] for r in sub])),
            "f1_8cls_std": float(np.std([r["f1_8cls"] for r in sub], ddof=1)) if len(sub) > 1 else 0.0,
            "accuracy_mean": float(np.mean([r["accuracy"] for r in sub])),
            "accuracy_std": float(np.std([r["accuracy"] for r in sub], ddof=1)) if len(sub) > 1 else 0.0,
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in sub])),
            "best_epoch_min": int(np.min([r["best_epoch"] for r in sub])),
            "best_epoch_max": int(np.max([r["best_epoch"] for r in sub])),
            "best_C_mode": float(max(set(r["best_C"] for r in sub),
                                     key=[r["best_C"] for r in sub].count)),
        }
        out.append(rec)
    cols = list(out[0].keys())
    with open(os.path.join(OUT, "screening_agg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"[OK] screening_agg.csv ({len(out)} points modèle×fraction)")
    return out


def write_per_class_by_regime(rows):
    keys = sorted({(r["model"], r["fraction"]) for r in rows})
    recs = []
    for model, frac in keys:
        sub = [r for r in rows if r["model"] == model and r["fraction"] == frac]
        rec = {"model": model, "display": sub[0]["display"], "fraction": frac,
               "n_train_tiles": sub[0]["n_train_tiles"], "n_seeds": len(sub)}
        for c in CLASSES_11:
            v = [r[f"f1_{c}"] for r in sub]
            rec[c] = float(np.mean(v))
            rec[f"{c}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        recs.append(rec)
    cols = list(recs[0].keys())
    with open(os.path.join(OUT, "per_class_by_regime.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)
    print(f"[OK] per_class_by_regime.csv ({len(recs)} lignes)")


def write_per_class_all_models(rows):
    """F1 par classe de tous les modèles à 100 % : gelés (probe canonique) + FT (probe interne)."""
    recs = []
    pk = json.load(open(p("results", "without_rhol", "probe_knn_cgrid.json")))["probe"]
    for key, (display, dim) in FROZEN_MODELS.items():
        if key not in pk:
            print(f"  [SKIP] {key} absent de probe_knn_cgrid.json")
            continue
        e = pk[key]
        rec = {"model": key, "display": display, "type": "frozen", "dim": dim,
               "n_seeds": 1, "probe": "canonique_cgrid",
               "f1_pres": e["test"]["f1_macro_pres"], "accuracy": e["test"]["accuracy"],
               "f1_weighted": e["test"]["f1_weighted"], "best_C": e["best_C"]}
        for c in CLASSES_11:
            rec[c] = e["f1_per_class_test"].get(c, float("nan"))
        recs.append(rec)

    for key in RUN_FAMILIES:
        sub = [r for r in rows if r["model"] == key and r["fraction"] == 1.0]
        if not sub:
            continue
        rec = {"model": key, "display": sub[0]["display"], "type": sub[0]["type"],
               "dim": 2048 if "resnet" in key else 768, "n_seeds": len(sub),
               "probe": "interne_run",
               "f1_pres": float(np.mean([r["f1_pres"] for r in sub])),
               "accuracy": float(np.mean([r["accuracy"] for r in sub])),
               "f1_weighted": float("nan"),
               "best_C": max(set(r["best_C"] for r in sub),
                             key=[r["best_C"] for r in sub].count)}
        for c in CLASSES_11:
            rec[c] = float(np.mean([r[f"f1_{c}"] for r in sub]))
        recs.append(rec)

    recs.sort(key=lambda r: -r["f1_pres"])
    cols = ["model", "display", "type", "dim", "n_seeds", "probe", "f1_pres",
            "accuracy", "f1_weighted", "best_C"] + CLASSES_11
    with open(os.path.join(OUT, "per_class_all_models.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] per_class_all_models.csv ({len(recs)} modèles)")


def checks(agg):
    """Contrôles de non-régression contre le tableau actuel de datacurves.tex."""
    exp = [("vitb16_full_old", 1.0, 0.4747, 0.0015),
           ("vitb16_full_old", 0.01, 0.4157, 0.0147),
           ("vitb16_mhsa_old", 0.7, 0.4783, 0.0041),
           ("vitb16_scratch_old", 1.0, 0.3549, 0.0110),
           ("dinov3_vitb16_lvd_lora_r8", 1.0, 0.482865, 0.002966),
           ("dinov3_vitb16_lvd_lora_r8", 0.005, 0.40772, 0.008399)]
    ok = True
    for model, frac, f1, sd in exp:
        r = [a for a in agg if a["model"] == model and abs(a["fraction"] - frac) < 1e-9]
        if not r:
            print(f"  [FAIL] {model} @ {frac} absent"); ok = False; continue
        d = abs(r[0]["f1_pres_mean"] - f1)
        status = "OK  " if d < 5e-4 else "FAIL"
        if d >= 5e-4:
            ok = False
        print(f"  [{status}] {model:28s} {frac:5.3f}  "
              f"attendu {f1:.4f}±{sd:.4f}  obtenu {r[0]['f1_pres_mean']:.4f}"
              f"±{r[0]['f1_pres_std']:.4f}")
    return ok


def main():
    ensure_out()
    print("== Lecture des runs ==")
    rows = load_runs()
    print(f"\nTotal : {len(rows)} runs\n")
    write_raw(rows)
    agg = write_agg(rows)
    write_per_class_by_regime(rows)
    write_per_class_all_models(rows)
    print("\n== Contrôles de non-régression ==")
    ok = checks(agg)
    print("\n[RÉSULTAT]", "tous les contrôles passent" if ok else "ÉCHEC — voir ci-dessus")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
