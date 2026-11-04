#!/usr/bin/env python3
"""GATv2 sur le graphe complet des tuiles (annotées + auxiliaires), features gelées.

Protocole : mêmes features, même split canonique 11 classes, même métrique (F1 macro),
mêmes 3 seeds, même unité de test que les autres bras. Seule change la mécanique
d'agrégation.

Bras :
  MEAN9    moyenne des 9 voisins du graphe COMPLET (auxiliaires inclus), puis sonde
           canonique — c'est le comparateur qui reçoit exactement la même information.
  GNN2     GATv2, 2 couches (champ réceptif ±224 px), features d'arête = décalage
  GNN3     GATv2, 3 couches (champ réceptif ±336 px)
  GNN2rw   contrôle : mêmes arêtes en nombre, sources permutées (appariement rompu)

Budget de paramètres volontairement aligné sur L1/L2 (~170-200 k) pour que la
comparaison porte sur la mécanique, pas sur la capacité.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 112
OFFS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)]
N_OFF = len(OFFS)
N_CLS = 11


# --------------------------------------------------------------------------- graphe
def build_graph(keys_lab, lab_pos, aux_pos, rewire_seed: int | None = None):
    """Retourne node_index, edge_index (2,E), edge_attr (E,), n_nodes."""
    pos = {}
    for i, p in enumerate(lab_pos):
        pos[p] = i
    n_lab = len(lab_pos)
    for j, p in enumerate(aux_pos):
        if p not in pos:
            pos[p] = n_lab + j
    n_nodes = n_lab + len(aux_pos)
    src, dst, attr = [], [], []
    for p, i in pos.items():
        o, r, c = p
        for k, (dr, dc) in enumerate(OFFS):
            q = (o, r + STRIDE * dr, c + STRIDE * dc)
            j = pos.get(q)
            if j is not None:
                src.append(j)
                dst.append(i)
                attr.append(k)
    src = np.array(src, dtype=np.int64)
    dst = np.array(dst, dtype=np.int64)
    attr = np.array(attr, dtype=np.int64)
    if rewire_seed is not None:
        rng = np.random.RandomState(rewire_seed)
        for k in range(N_OFF):
            m = attr == k
            src[m] = rng.permutation(src[m])
    return pos, n_nodes, src, dst, attr


# --------------------------------------------------------------------------- modèle
class GATv2Layer(nn.Module):
    def __init__(self, d: int, n_off: int = N_OFF):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.V = nn.Linear(d, d, bias=False)
        self.e = nn.Embedding(n_off, d)
        self.a = nn.Linear(d, 1, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, dst, src, eattr, n_nodes):
        h = self.W(x)
        v = self.V(x)
        m = v[src] + self.e(eattr)
        s = F.leaky_relu(self.a(m), 0.2).squeeze(-1)
        mx = torch.full((n_nodes,), -1e9, device=x.device).index_reduce_(0, dst, s, "amax")
        ex = torch.exp(s - mx[dst])
        den = torch.zeros(n_nodes, device=x.device).index_add_(0, dst, ex)
        alpha = ex / (den[dst] + 1e-9)
        out = torch.zeros_like(v).index_add_(0, dst, alpha.unsqueeze(-1) * v[src])
        return self.norm(x + h + out)


class GATv2(nn.Module):
    def __init__(self, din: int, d: int, layers: int, n_cls: int = N_CLS, p: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(din, d), nn.LayerNorm(d), nn.Dropout(p))
        self.layers = nn.ModuleList([GATv2Layer(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d, n_cls))

    def forward(self, x, dst, src, eattr, n_nodes):
        h = self.inp(x)
        for l in self.layers:
            h = l(h, dst, src, eattr, n_nodes)
        return self.head(h)


# --------------------------------------------------------------------------- métriques
def f1_macro(y, p, k=N_CLS):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    return float(f[~np.isnan(f)].mean())


def macro_f1_torch(y, logits):
    p = logits.argmax(1)
    return f1_macro(y.detach().cpu().numpy(), p.detach().cpu().numpy())


# --------------------------------------------------------------------------- entraînement
def train_gnn(X, y, tr, va, te, dst, src, eattr, n_nodes, layers, seed, epochs, d, lr,
              wd=0.05, p=0.1):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    # permutation de l'ordre des arêtes (les agrégats index_add_ sont insensibles à l'ordre)
    x = torch.from_numpy(X)
    model = GATv2(X.shape[1], d, layers, p=p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 2))
    yt = torch.from_numpy(y)
    ytr, yva = yt[tr], yt[va]
    idx_tr = torch.from_numpy(tr)
    best = {"val": -1, "state": None, "ep": -1}
    bad = 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, dst, src, eattr, n_nodes)
        loss = F.cross_entropy(logits[idx_tr], ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, dst, src, eattr, n_nodes)
            vf1 = macro_f1_torch(yva, logits[torch.from_numpy(va)])
        if vf1 > best["val"]:
            best = {"val": vf1, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "ep": ep}
            bad = 0
        else:
            bad += 1
            if bad >= 15:
                break
        if ep % 5 == 0:
            print(f"    ep={ep} loss={loss.item():.4f} valF1={vf1:.4f}", flush=True)
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        logits = model(x, dst, src, eattr, n_nodes)
        teF1 = macro_f1_torch(yt[te], logits[torch.from_numpy(te)])
        pred = logits[torch.from_numpy(te)].argmax(1).numpy().astype(np.uint8)
    return teF1, best["val"], best["ep"], pred, sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- sonde moyenne
REPO = str(Path(__file__).resolve().parents[3])


def probe_mean(Xmean, y, tr, va, te):
    """Sonde canonique sur la moyenne du voisinage complet (comparateur non appris)."""
    sys.path.insert(0, REPO)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    sc = StandardScaler().fit(Xmean[tr])
    Xs = sc.transform(Xmean).astype(np.float32)
    best = (-1, None, None)
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        clf = make_canonical_lr(C=C)
        clf.fit(Xs[tr], y[tr])
        v = f1_macro(y[va], clf.predict(Xs[va]))
        if v > best[0]:
            best = (v, C, clf)
    v, C, clf = best
    p = clf.predict(Xs[te]).astype(np.uint8)
    return f1_macro(y[te], p), v, C, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--features", required=True, help="features_tile.npy des tuiles annotées")
    ap.add_argument("--aux-features", default="")
    ap.add_argument("--aux-index", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="MEAN9,GNN2,GNN3,GNN2rw")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()

    global REPO
    REPO = a.repo
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    tiles = json.load(open(a.split))["tiles"]
    X = np.load(a.features).astype(np.float32)
    assert X.shape[0] == len(tiles), (X.shape, len(tiles))
    y = np.array([t["label"] for t in tiles], dtype=np.int64)
    sp = np.array([t["split"] for t in tiles])
    lab_pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]

    aux_X = np.zeros((0, X.shape[1]), np.float32)
    aux_pos = []
    if a.aux_features and os.path.exists(a.aux_features):
        aux_X = np.load(a.aux_features).astype(np.float32)
        aux_pos = [(d["ortho"], d["row"], d["col"]) for d in json.load(open(a.aux_index))]
        assert len(aux_pos) == aux_X.shape[0]
    print(f"[gnn] annotées={len(tiles)} auxiliaires={len(aux_pos)} dim={X.shape[1]}", flush=True)

    pos, n_nodes, src, dst, eattr = build_graph(None, lab_pos, aux_pos)
    print(f"[gnn] nœuds={n_nodes} arêtes={len(src)} (±{STRIDE}px, 8 voisins)", flush=True)

    Xall = np.concatenate([X, aux_X], axis=0)
    tr = np.where(sp == "train")[0]
    va = np.where(sp == "val")[0]
    te = np.where(sp == "test")[0]
    # standardisation identique à celle des sondes (fit sur le train labellisé uniquement),
    # appliquée aussi aux auxiliaires : sinon le GNN part avec des features à échelle brute
    mu = Xall[tr].mean(0)
    sd = Xall[tr].std(0) + 1e-6
    Xall = ((Xall - mu) / sd).astype(np.float32)

    dst_t = torch.from_numpy(dst)
    src_t = torch.from_numpy(src)
    eattr_t = torch.from_numpy(eattr)
    results, preds = {}, {}

    if "MEAN9" in arms:
        t0 = time.time()
        # moyenne des voisins présents sur le graphe complet (inclut la tuile elle-même)
        acc = np.zeros((n_nodes, Xall.shape[1]), np.float32)
        cnt = np.zeros((n_nodes, 1), np.float32)
        for k in range(N_OFF):
            m = eattr == k
            acc[dst[m]] += Xall[src[m]]
            cnt[dst[m]] += 1
        acc[np.arange(len(tiles))] += X  # la tuile elle-même (déjà dans src ? non : centre exclu)
        cnt[np.arange(len(tiles))] += 1
        mean = acc / np.maximum(cnt, 1)
        f1, vf1, C, p = probe_mean(mean[: len(tiles)], y, tr, va, te)
        results["MEAN9"] = {"f1_macro_pres": f1, "val_f1": vf1, "best_C": C,
                            "n_params": 0, "seconds": round(time.time() - t0, 1)}
        preds["MEAN9"] = p
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
        np.save(os.path.join(a.out_dir, "preds", "MEAN9.npy"), p)
        json.dump({"arms": results}, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=1)
        print(f"[MEAN9] testF1={f1:.4f} (C={C})", flush=True)

    for arm in [x for x in arms if x.startswith("GNN")]:
        layers = int(arm[3]) if len(arm) > 3 and arm[3].isdigit() else 2
        rw = arm.endswith("rw")
        if rw:
            _, _, src_r, dst_r, ea_r = build_graph(None, lab_pos, aux_pos, rewire_seed=123)
            S, D, E = torch.from_numpy(src_r), torch.from_numpy(dst_r), torch.from_numpy(ea_r)
        else:
            S, D, E = src_t, dst_t, eattr_t
        t0 = time.time()
        f1s, preds_seeds, ep_best, npar = [], [], [], 0
        for s in seeds:
            f1, vf1, ep, p, npar = train_gnn(Xall, y, tr, va, te, D, S, E, n_nodes,
                                             layers, s, a.epochs, a.hidden, a.lr,
                                             a.wd, a.dropout)
            f1s.append(f1)
            preds_seeds.append(p)
            ep_best.append(ep)
            print(f"[{arm}] seed={s} testF1={f1:.4f} valF1={vf1:.4f} ep={ep} "
                  f"params={npar} ({time.time() - t0:.0f}s)", flush=True)
        results[arm] = {"f1_macro_pres": float(np.mean(f1s)),
                        "f1_macro_pres_std": float(np.std(f1s)),
                        "seed_f1": [float(x) for x in f1s], "best_epochs": ep_best,
                        "n_params": int(npar), "layers": layers,
                        "seconds": round(time.time() - t0, 1)}
        preds[arm] = np.stack(preds_seeds)
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
        np.save(os.path.join(a.out_dir, "preds", f"{arm}.npy"), preds[arm])
        json.dump({"arms": results, "n_nodes": n_nodes, "n_edges": int(len(src)),
                   "n_aux": len(aux_pos), "n_tiles": len(tiles), "hidden": a.hidden,
                   "epochs": a.epochs, "seeds": seeds},
                  open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=1)
        print(f"[{arm}] testF1={np.mean(f1s):.4f}±{np.std(f1s):.4f}", flush=True)

    os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
    for k, v in preds.items():
        np.save(os.path.join(a.out_dir, "preds", f"{k}.npy"), v)
    json.dump({"arms": results, "n_nodes": n_nodes, "n_edges": int(len(src)),
               "n_aux": len(aux_pos), "n_tiles": len(tiles), "hidden": a.hidden,
               "epochs": a.epochs, "seeds": seeds},
              open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
