#!/usr/bin/env python3
"""CLEAN baseline probe — ONE probe, frozen protocol.

Standalone & namespaced to results/baseline_clean_2026-10-08/. Never imports the
historical pipeline. Single-thread BLAS is forced BEFORE numpy/sklearn import.

Protocol (frozen): StandardScaler(train) -> LR(lbfgs, max_iter=2000, seed=42),
C grid [1e-4,1e-3,1e-2,1e-1,1,10], best_C by FULL-TRAIN validation f1_macro_all,
deterministic tie-break = LARGER C; refit on full train; report f1_macro_pres and
f1_macro_all on test. No 20k subsample. Schema inferred from labels.

Usage: python3 run_baseline_probe.py --id frozen__dinov3_vitb16_lvd
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = HERE
ROOT = os.path.dirname(os.path.dirname(BASE))  # .../Mémoire
PROBES = os.path.join(BASE, "probes")
PREDS = os.path.join(BASE, "preds")

CLASSES_12 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS",
              "PETF", "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SEED = 42
MAX_ITER = 2000


def _files(entry):
    if entry["kind"] == "flat":
        b = os.path.join(ROOT, entry["path"])
        return {s: (f"{b}_{s}.npy", f"{b}_{s}_labels.npy") for s in ("train", "val", "test")}
    b = os.path.join(ROOT, entry["path"])
    return {s: (os.path.join(b, f"{s}.npy"), os.path.join(b, f"{s}_labels.npy")) for s in ("train", "val", "test")}


def infer_schema(ytr, yva, yte):
    ymax = int(max(np.max(ytr), np.max(yva), np.max(yte)))
    return "12cls" if ymax >= 11 else "11cls"


def cascade(X, y, names, drop_names):
    drop = sorted((names.index(c) for c in drop_names if c in names), reverse=True)
    y2 = y.copy()
    for d in drop:
        m = y2 != d
        X, y2 = X[m], y2[m]
        y2[y2 > d] -= 1
    return X, y2


def _lr(c):
    from sklearn.linear_model import LogisticRegression
    kw = dict(C=float(c), solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)
    try:
        return LogisticRegression(multi_class="multinomial", **kw)
    except TypeError:
        return LogisticRegression(**kw)


def run_probe(entry, manifest=None):
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import sklearn
    t0 = time.time()
    files = _files(entry)
    X = {}; y = {}
    for s in ("train", "val", "test"):
        X[s] = np.load(files[s][0]).astype(np.float32)
        y[s] = np.load(files[s][1]).ravel().astype(np.int64)
    src = infer_schema(y["train"], y["val"], y["test"])
    names = CLASSES_12 if src == "12cls" else [c for c in CLASSES_12 if c != "RHOL"]
    target = ("RHOL",) if src == "12cls" else ()           # 11-class target
    tnames = [c for c in names if c not in target]
    X["train"], y["train"] = cascade(X["train"], y["train"], names, target)
    X["val"], y["val"] = cascade(X["val"], y["val"], names, target)
    X["test"], y["test"] = cascade(X["test"], y["test"], names, target)

    sc = StandardScaler()
    Xtr = sc.fit_transform(X["train"])
    Xva = sc.transform(X["val"])
    Xte = sc.transform(X["test"])
    del X

    labels_all = list(range(len(tnames)))
    grid = []
    for c in C_GRID:
        clf = _lr(c).fit(Xtr, y["train"])
        f1v = float(f1_score(y["val"], clf.predict(Xva), average="macro",
                             zero_division=0, labels=labels_all))
        grid.append({"C": c, "val_f1_macro_all": f1v})
    # deterministic tie-break: larger C wins exact ties
    best = max(grid, key=lambda g: (g["val_f1_macro_all"], g["C"]))
    ordered = sorted(grid, key=lambda g: (g["val_f1_macro_all"], g["C"]), reverse=True)
    margin = ordered[0]["val_f1_macro_all"] - (ordered[1]["val_f1_macro_all"] if len(ordered) > 1 else 0.0)

    clf = _lr(best["C"]).fit(Xtr, y["train"])
    yp = clf.predict(Xte)
    yte = y["test"]
    present = sorted({int(v) for v in yte})
    per_class = f1_score(yte, yp, average=None, zero_division=0, labels=labels_all)

    def hashes():
        if not manifest:
            return {}
        out = {}
        for s in ("train", "val", "test"):
            for path in files[s]:
                rel = os.path.relpath(path, ROOT)
                if rel in manifest:
                    out[rel] = manifest[rel].get("sha256")
        return out

    res = {
        "id": entry["id"], "key": entry["key"], "family": entry["family"],
        "path": entry["path"], "kind": entry["kind"], "seed": entry["seed"], "fraction": entry["fraction"],
        "deployable": entry["deployable"], "claim_tier": entry["claim_tier"],
        "source_schema": src, "n_classes": len(tnames), "classes": tnames,
        "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]), "n_test": int(Xte.shape[0]),
        "dim": int(Xtr.shape[1]),
        "grid": grid, "best_C": best["C"], "best_val_f1_macro_all": best["val_f1_macro_all"],
        "margin": float(margin), "tie_break_rule": "exact tie -> larger C",
        "test": {
            "f1_macro_pres": float(f1_score(yte, yp, average="macro", zero_division=0, labels=present)),
            "f1_macro_all": float(f1_score(yte, yp, average="macro", zero_division=0, labels=labels_all)),
            "f1_weighted": float(f1_score(yte, yp, average="weighted", zero_division=0, labels=labels_all)),
            "accuracy": float(accuracy_score(yte, yp)),
        },
        "per_class_f1_test": {c: float(v) for c, v in zip(tnames, per_class)},
        "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
        "git_rev": _git(), "embedding_hashes": hashes(),
        "runtime_s": round(time.time() - t0, 1),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return res, yte, yp


def _git():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _atomic_npz(path, yte, yp, best_C):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, y_true=yte.astype(np.int16), y_pred=yp.astype(np.int16),
                        best_C=np.array([best_C], dtype=np.float64))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(BASE, "model_map.json"))
    ap.add_argument("--id", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    mp = json.load(open(args.map))
    entry = next((p for p in mp["probes"] if p["id"] == args.id), None)
    if entry is None:
        print(f"[ERR] unknown probe id {args.id}", file=sys.stderr); sys.exit(2)
    out = os.path.join(PROBES, f"{args.id}.json")
    if os.path.exists(out) and not args.force:
        print(f"[skip] {args.id}"); return
    man = None
    mpath = os.path.join(BASE, "embedding_manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
    res, yte, yp = run_probe(entry, man)
    _atomic_npz(os.path.join(PREDS, f"{args.id}.npz"), yte, yp, res["best_C"])
    _atomic_json(out, res)
    print(f"[ok] {args.id} C={res['best_C']:g} margin={res['margin']:.4f} "
          f"f1pres={res['test']['f1_macro_pres']:.4f} f1all={res['test']['f1_macro_all']:.4f} "
          f"({res['runtime_s']}s)")


if __name__ == "__main__":
    main()
