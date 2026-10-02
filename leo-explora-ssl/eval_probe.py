#!/usr/bin/env python3
"""Probe comparatif GELÉ vs ADAPTÉ (local, CPU) : qui gagne en AUPRC ?

Protocole (identique à night_exp4_select : apparié, holdout spatial par
blocs de 20m, mêmes hyperparamètres des deux côtés) :
  1. --dump : construit probe_labels.json {key: {f, label, x, y}} depuis
     bank.npz (positifs, x/y UTM) + tile_labels*.csv (idx -> manifest).
     SOL -> négatif, MULTI/SKIP -> négatif.
  2. transfère probe_labels.json + ssl_tiles sur Narval, extrait 2x :
     - probe_emb_frozen.npz (ckpt SimDINOv2 d'origine)
     - probe_emb_adapted.npz (last.pth ExPLoRA)
     via extract_probe_embeddings.py
  3. --probe : régression logistique par espèce, AUPRC holdout, comparatif.

Usage :
    python eval_probe.py --dump --leo /home/erazal/annotations_leo --out probe_labels.json
    # ... extraction Narval (voir README) ...
    python eval_probe.py --probe --labels probe_labels.json \\
        --frozen probe_emb_frozen.npz --adapted probe_emb_adapted.npz
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

BASE_LEO = Path("/home/erazal/annotations_leo")
BLOCK_M = 20.0
SPECIES = ["Lotcorn", "Leuvul", "Ascsyr", "Daucar", "Eumac", "Solcan"]
CSV_BY_SITE = {
    "clairiere": "tile_labels.csv",
    "maison": "tile_labels_maison.csv",
    "trail": "tile_labels_trail.csv",
    "rousseau": "tile_labels_rousseau.csv",
}
RASTER = {
    "clairiere": "Orthom_Clairiere_9Aout23_WGS84UTM18N.tif",
}


def dump(leo: Path, out: Path):
    D = Path("/home/erazal/Documents/Mémoire/Dataset_Leo/Orthomosaiques")
    man = {}
    for site in ["clairiere", "maison", "trail", "rousseau"]:
        p = leo / f"data/ssl_manifest_{site}.json"
        if p.exists():
            man[site] = json.loads(p.read_text())["tiles"]
        else:  # fallback manifests d'affichage (mêmes idx/fenêtres)
            man[site] = json.loads(
                (BASE_LEO / "web" / f"manifest_{site}.json").read_text())["tiles"]
    labels: dict[str, dict] = {}
    # 1) bank = positifs avec coords (100% clairière, vérifié)
    bank = np.load(leo / "embeddings" / "bank.npz", allow_pickle=True)
    import rasterio
    from rasterio import transform as rio_t
    src = rasterio.open(D / RASTER["clairiere"])
    H, W = src.height, src.width
    n_tc = W // 224
    for sp, x, y in zip(bank["species"].astype(str), bank["x"], bank["y"]):
        r, c = rio_t.rowcol(src.transform, float(x), float(y))
        tr, tc = int(r) // 224, int(c) // 224
        if not (0 <= tr < H // 224 and 0 <= tc < n_tc):
            continue
        idx = tr * n_tc + tc
        t = man["clairiere"].get(str(idx))
        if t is None:
            continue
        key = f"bank_{sp}_{idx}"
        labels[key] = {"f": f"clairiere/{idx}.jpg",
                       "label": str(sp), "x": float(x), "y": float(y)}
    src.close()
    # 2) tile_labels CSV (coords = centre approximatif via tr/tc -> x/y path)
    for site, cf in CSV_BY_SITE.items():
        p = leo / cf
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                idx, lab = int(row["idx"]), row["label"]
                t = man[site].get(str(idx))
                if t is None:
                    continue
                key = f"tile_{site}_{idx}"
                labels[key] = {"f": f"{site}/{idx}.jpg",
                               "label": lab, "x": None, "y": None,
                               "tr": t["tr"], "tc": t["tc"], "site": site}
    # x/y des tuiles via grille clairiere (seul site avec bank/coords ;
    # les autres sites servent de négatifs hors-holdout)
    print(f"labels: {len(labels)} ({Counter(v['label'] for v in labels.values())})")
    out.write_text(json.dumps(labels))
    print("écrit:", out)


def block_id(x, y):
    return (np.asarray(x) // BLOCK_M).astype(int) * 1000000 + \
           (np.asarray(y) // BLOCK_M).astype(int)


def probe(labels_p, frozen_p, adapted_p):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    lab = json.loads(Path(labels_p).read_text())
    fr = np.load(frozen_p)
    ad = np.load(adapted_p)
    fmap = {str(k): v for k, v in zip(fr["key"], fr["emb"])}
    amap = {str(k): v for k, v in zip(ad["key"], ad["emb"])}

    keys = [k for k in lab if k in fmap and k in amap]
    print(f"{len(keys)}/{len(lab)} échantillons communs")
    rng = np.random.default_rng(0)
    print(f"{'espèce':8s} {'n_pos':>6s} {'frozen':>8s} {'adapté':>8s} {'Δ':>8s}")
    for sp in SPECIES:
        pos = [k for k in keys if lab[k]["label"] == sp]
        neg = [k for k in keys if lab[k]["label"] not in SPECIES]
        if len(pos) < 5 or len(neg) < 20:
            print(f"{sp:8s} {len(pos):6d}   (trop peu de données)")
            continue
        # holdout spatial sur les positifs à coords connues
        xc = np.array([lab[k].get("x") or 0 for k in pos])
        yc = np.array([lab[k].get("y") or 0 for k in pos])
        known = (xc != 0)
        if known.sum() >= 8:
            bids = block_id(xc[known], yc[known])
            ubl = np.unique(bids)
            rng.shuffle(ubl)
            hold = set(ubl[:max(1, len(ubl) // 4)])
            te_pos = [k for k, b in zip(np.array(pos)[known], bids) if b in hold]
            tr_pos = [k for k in pos if k not in te_pos]
        else:
            tr_pos, te_pos = pos[:len(pos) * 3 // 4], pos[len(pos) * 3 // 4:]
        te_neg = rng.choice(neg, size=min(2000, len(neg) // 2), replace=False).tolist()
        tr_neg = [k for k in neg if k not in te_neg][:5000]
        res = {}
        for name, mp in (("frozen", fmap), ("adapté", amap)):
            Xtr = np.stack([mp[k] for k in tr_pos + tr_neg])
            ytr = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
            Xte = np.stack([mp[k] for k in te_pos + te_neg])
            yte = np.array([1] * len(te_pos) + [0] * len(te_neg))
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.003, max_iter=1000)
            clf.fit(sc.transform(Xtr), ytr)
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            res[name] = average_precision_score(yte, p)
        d = res["adapté"] - res["frozen"]
        print(f"{sp:8s} {len(pos):6d} {res['frozen']:8.3f} {res['adapté']:8.3f} "
              f"{d:+8.3f} {'✅' if d > 0.02 else ('❌' if d < -0.02 else '≈')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--leo", default=str(BASE_LEO))
    ap.add_argument("--out", default="probe_labels.json")
    ap.add_argument("--labels", default="probe_labels.json")
    ap.add_argument("--frozen", default="probe_emb_frozen.npz")
    ap.add_argument("--adapted", default="probe_emb_adapted.npz")
    args = ap.parse_args()
    if args.dump:
        dump(Path(args.leo), Path(args.out))
    if args.probe:
        probe(args.labels, args.frozen, args.adapted)


if __name__ == "__main__":
    main()
