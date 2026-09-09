#!/usr/bin/env python3
"""Sweep tile-only : lbfgs canonique vs lin-AdamW vs MLP-2 sur les 33 groupes du palier.

Extension de `fusion_head_sweep.py` à la population ENTIÈRE du benchmark (les
groupes de `significance_tier.GROUPS` — embeddings tuile seule, locaux, déjà en
schéma 11 classes pour les runs ft ; RHOL retirée pour les gelés).

Pourquoi
--------
Juillet : `mlp_probe.py` (20 modèles) a trouvé non-linéarité ≈ +0,003 médiane,
0/20 au-dessus de 0,01. Depuis, 13 bras Stage A + ViT-S LoRA + contextes sont
entrés au palier. Ce script remet les 33 groupes à l'épreuve de la même question,
avec les MÊMES conventions (sélection val, 3 seeds de training, delta vs
lin_adamw) — et refait le lbfgs canonique au passage : ses valeurs doivent
retrouver `tier_preds_cache` (contrôle de cohérence gratuit).

Têtes : lin_lbfgs (canonique), lin_adamw (contrôle optimiseur), mlp2.
(bil_diag/film sans sens sur vue unique — ils restent pour le sweep fused.)

Usage local (le runner fait ça) :
    python3 scripts/tile_head_sweep.py --group "SimDINOv2-B NormTuning"
    python3 scripts/tile_head_sweep.py --all           # liste les groupes
Sorties : results/rapport_data/tile_heads/<slug(groupe)>.json + _aggregate.csv
Idempotent : JSON existant sauté sauf --force.
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import argparse
import csv
import json
import re
import sys
import time

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(_HERE, "rapport"))   # significance_tier, registry
sys.path.insert(0, _os.path.dirname(_HERE))            # racine (src/)

from significance_tier import GROUPS, C_GRID, f1_pres          # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402

TRAIN_SEEDS = (0, 1, 2)
# Grille réduite (16→8 configs vs fused) : 33 groupes, pas 24 tags — le coût
# total doit rester sous ~2 jours CPU parallélisés.
HP_MLP = [(H, p, wd) for H in (256, 512) for p in (0.2, 0.5) for wd in (0.0, 1e-4)]
OUT_DIR = _os.environ.get("TILE_HEAD_OUT", _os.path.join(
    _os.path.dirname(_HERE), "results", "rapport_data", "tile_heads"))
MAX_ITER_LBFGS = 2000
EPOCHS = int(_os.environ.get("TILE_HEAD_EPOCHS", "120"))


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _make_mlp(x_dim, n_cls, hp):
    import torch.nn as nn
    H, p, wd = hp

    class MLP2(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x_dim, H), nn.ReLU(), nn.Dropout(p),
                nn.Linear(H, H), nn.ReLU(), nn.Dropout(p),
                nn.Linear(H, n_cls))

        def forward(self, x):
            return self.net(x)
    return MLP2(), wd


def _train_head(xtr, ytr, xva, yva, xte, yte, hp, seed, epochs):
    """MLP2 AdamW, early stopping patience 15 sur val f1_macro_pres."""
    import torch
    torch.manual_seed(seed)
    model, wd = _make_mlp(xtr.shape[1], 11, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(xva, dtype=torch.float32)
    xe = torch.tensor(xte, dtype=torch.float32)
    best_val, best_state, bad = -1.0, None, 0
    n = xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            f1v = f1_pres(yva, model(xv).argmax(1).numpy())
        if f1v > best_val + 1e-5:
            best_val, bad = f1v, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return f1_pres(yte, model(xe).argmax(1).numpy()), best_val


def _linear_adamw(xtr, ytr, xva, yva, xte, yte, wd, seed, epochs):
    import torch
    torch.manual_seed(seed)
    model = torch.nn.Linear(xtr.shape[1], 11)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(xtr); yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(xva); xe = torch.tensor(xte)
    best_val, best_state, bad = -1.0, None, 0
    n = xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            f1v = f1_pres(yva, model(xv).argmax(1).numpy())
        if f1v > best_val + 1e-5:
            best_val, bad = f1v, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return f1_pres(yte, model(xe).argmax(1).numpy()), best_val


def sweep_group(name):
    from significance_tier import load_split
    entry = next(g for g in GROUPS if g[0] == name)
    _, kind, path, seeds, _ = entry
    per_seed = []
    for s in seeds:
        d = load_split(kind, path, s)
        sc = StandardScaler().fit(d["train"][0])
        feats = {sp: (sc.transform(d[sp][0]), d[sp][1]) for sp in ("train", "val", "test")}
        per_seed.append(feats)

    out = {"group": name, "kind": kind, "path": path, "n_seeds": len(seeds),
           "x_dim": int(per_seed[0]["train"][0].shape[1]), "fused": False}
    t0 = time.time()

    # lbfgs canonique (sélection C sur val f1_macro_all — miroir significance_tier)
    from sklearn.metrics import f1_score
    lb = {"test": [], "C": [], "val": []}
    for feats in per_seed:
        xtr, ytr = feats["train"]; xva, yva = feats["val"]; xte, yte = feats["test"]
        best = (-1, None)
        for C in C_GRID:
            lr = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER_LBFGS,
                                    random_state=42).fit(xtr, ytr)
            f1v = f1_score(yva, lr.predict(xva), average="macro", zero_division=0,
                           labels=list(range(11)))
            if f1v > best[0]:
                best = (f1v, C)
        lr = LogisticRegression(C=best[1], solver="lbfgs", max_iter=MAX_ITER_LBFGS,
                                random_state=42).fit(xtr, ytr)
        lb["test"].append(f1_pres(yte, lr.predict(xte)))
        lb["C"].append(best[1]); lb["val"].append(best[0])
    out["lin_lbfgs"] = {"f1_mean": float(np.mean(lb["test"])), "best_C": lb["C"]}

    # AdamW heads : sélection config sur val (moyenne seeds), test rapporté
    for head, fn in (("lin_adamw", None), ("mlp2", None)):
        grid = [(0, 0.0, 0.0)] if head == "lin_adamw" else HP_MLP
        per_cfg = {}
        for hp in grid:
            f1s, f1vs = [], []
            for i, feats in enumerate(per_seed):
                seed = TRAIN_SEEDS[i % 3]
                xtr, ytr = feats["train"]; xva, yva = feats["val"]; xte, yte = feats["test"]
                if head == "lin_adamw":
                    f1t, f1v = _linear_adamw(xtr, ytr, xva, yva, xte, yte,
                                             hp[2], seed, EPOCHS)
                else:
                    f1t, f1v = _train_head(xtr, ytr, xva, yva, xte, yte,
                                           hp, seed, EPOCHS)
                f1s.append(f1t); f1vs.append(f1v)
            per_cfg[repr(hp)] = {"val_mean": float(np.mean(f1vs)),
                                 "test": [float(x) for x in f1s]}
        best_cfg = max(per_cfg.items(), key=lambda kv: kv[1]["val_mean"])
        te = np.array(best_cfg[1]["test"])
        out[head] = {"best_hp": best_cfg[0], "f1_mean": float(te.mean()),
                     "f1_std": float(te.std(ddof=1)) if len(te) > 1 else 0.0,
                     "test_per_seed": te.tolist(),
                     "val_mean": best_cfg[1]["val_mean"]}
    out["delta_mlp2_vs_lin_adamw"] = round(out["mlp2"]["f1_mean"] - out["lin_adamw"]["f1_mean"], 4)
    out["delta_mlp2_vs_lin_lbfgs"] = round(out["mlp2"]["f1_mean"] - out["lin_lbfgs"]["f1_mean"], 4)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


def aggregate():
    rows = []
    for jp in sorted(_os.path.join(OUT_DIR, f) for f in _os.listdir(OUT_DIR)
                     if f.endswith(".json")):
        d = json.load(open(jp))
        for k in ("lin_lbfgs", "lin_adamw", "mlp2"):
            v = d.get(k)
            if not v:
                continue
            rows.append({"group": d["group"], "head": k,
                         "f1_mean": round(v["f1_mean"], 4),
                         "f1_std": round(v.get("f1_std", 0.0), 4),
                         "delta_vs_lin_adamw": (round(v["f1_mean"] - d["lin_adamw"]["f1_mean"], 4)
                                                if k not in ("lin_adamw", "lin_lbfgs") else ""),
                         "best": v.get("best_hp", str(v.get("best_C", "")))})
    with open(_os.path.join(OUT_DIR, "_aggregate.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*")
    ap.add_argument("--root", default=None,
                    help="racine alternative pour les chemins de GROUPS "
                         "(défaut : dépôt local). Sur Narval : "
                         "$SCRATCH/head_sweep_inputs")
    ap.add_argument("--all", action="store_true", help="liste les groupes et quitte")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.root:
        import registry
        registry.ROOT = args.root
    if args.all:
        print("\n".join(g[0] for g in GROUPS)); return
    _os.makedirs(OUT_DIR, exist_ok=True)
    names = args.group or []
    if not names:
        ap.error("--group ... requis (voir --all)")
    for name in names:
        out_path = _os.path.join(OUT_DIR, slug(name) + ".json")
        if _os.path.exists(out_path) and not args.force:
            print(f"[tile-heads] skip {name}"); continue
        try:
            res = sweep_group(name)
        except Exception as e:
            print(f"[tile-heads] ERROR {name}: {e}", flush=True); continue
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[tile-heads] OK {name}  lbfgs={res['lin_lbfgs']['f1_mean']:.4f} "
              f"adamw={res['lin_adamw']['f1_mean']:.4f} mlp2={res['mlp2']['f1_mean']:.4f} "
              f"d_nl={res['delta_mlp2_vs_lin_adamw']:+.4f} ({res['runtime_s']}s)", flush=True)
    aggregate()


if __name__ == "__main__":
    main()
