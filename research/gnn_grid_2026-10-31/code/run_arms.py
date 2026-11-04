#!/usr/bin/env python3
"""Bras expérimentaux du pilote « agrégation relationnelle vs contexte fixe ».

Lit les features extraites (SimDINOv2 ViT-B/16) + le split pilote gelé, construit la grille
k×k depuis (ortho, tile_row_off, tile_col_off), entraîne les agrégateurs et exécute la sonde
canonique. Écrit un JSON + les prédictions de test par bras, puis le bootstrap hiérarchique.

Conforme à PRE_REGISTRATION.md (bras, seuils, métriques). Aucune sélection sur le test.

Usage :
  python run_arms.py --split out/pilot_split.json --features-dir out/features \
      --out-dir out --arms R0,R1,R2,R3,R4,L1,L2,L1p --seeds 0,1,2 --boot 10000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# BLAS mono-thread AVANT tout import numpy/sklearn (règle AGENTS.md §4.8)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RUBC", "SEDG", "TUSS", "WILL"]
N_CLASSES = 11
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
STRIDE = 112


# ----------------------------------------------------------------------------- métriques
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    """F1-macro (zero_division=0), labels fixés par défaut à 0..10 (identique à sklearn)."""
    if labels is None:
        labels = np.arange(N_CLASSES)
    cm = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).astype(np.float64)
    tp = np.diag(cm)[labels]
    fp = cm.sum(0)[labels] - tp
    fn = cm.sum(1)[labels] - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(f1.mean())


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    present = sorted({int(v) for v in y_true})
    return {
        "f1_macro_pres": float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                                        labels=present)),
        "f1_macro_all": float(f1_score(y_true, y_pred, average="macro", zero_division=0,
                                       labels=list(range(N_CLASSES)))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {c: float(v) for c, v in zip(
            CLASS_NAMES_11,
            f1_score(y_true, y_pred, average=None, zero_division=0,
                     labels=list(range(N_CLASSES))))},
        "n_test": int(len(y_true)),
        "classes_present": present,
    }


# ----------------------------------------------------------------------------- données
def load_all(split_path: str, feat_dir: str):
    split = json.load(open(split_path))
    idx = json.load(open(os.path.join(feat_dir, "index.json")))
    keys_split = [t["key"] for t in split["tiles"]]
    keys_feat = idx["keys"]
    assert keys_split == keys_feat, (
        "ordre des clés divergent entre le split et les features — refus d'apparier par ordre "
        f"implicite ({keys_split[:2]} vs {keys_feat[:2]})")
    n = len(keys_split)
    ortho_names = sorted({t["ortho"] for t in split["tiles"]})
    ortho_of = np.array([ortho_names.index(t["ortho"]) for t in split["tiles"]], dtype=np.int32)
    row = np.array([t["tile_row_off"] for t in split["tiles"]], dtype=np.int64)
    col = np.array([t["tile_col_off"] for t in split["tiles"]], dtype=np.int64)
    y = np.array([t["label"] for t in split["tiles"]], dtype=np.int64)
    sp = np.array([SPLIT_CODE[t["split"]] for t in split["tiles"]], dtype=np.int8)
    X = {}
    for v in idx["views"]:
        X[v] = np.load(os.path.join(feat_dir, f"features_{v}.npy")).astype(np.float32)
        assert X[v].shape == (n, idx["dim"]), (v, X[v].shape)
    meta = {"n": n, "ortho_names": ortho_names, "ortho": ortho_of, "row": row,
            "col": col, "y": y, "split": sp, "X": X, "index_meta": idx}
    return meta


def build_neighbors(ortho, row, col, k: int) -> np.ndarray:
    """Table (N, k*k) d'indices ; -1 = voisin absent (bord d'ortho ou de tuilage)."""
    pos = {(int(o), int(r), int(c)): i
           for i, (o, r, c) in enumerate(zip(ortho, row, col))}
    assert len(pos) == len(ortho), "coordonnées dupliquées : grille ambiguë"
    r = k // 2
    offs = [(i, j) for i in range(-r, r + 1) for j in range(-r, r + 1)]
    nb = np.full((len(ortho), k * k), -1, dtype=np.int32)
    for i, (o, rr, cc) in enumerate(zip(ortho, row, col)):
        for s, (di, dj) in enumerate(offs):
            nb[i, s] = pos.get((int(o), int(rr) + STRIDE * di, int(cc) + STRIDE * dj), -1)
    assert (nb[:, (k * k) // 2] == np.arange(len(ortho))).all(), "centre ≠ tuile elle-même"
    return nb


def grid_stats(nb: np.ndarray) -> dict:
    full = (nb >= 0).sum(1)
    return {
        "n_tiles": int(len(nb)),
        "k": int(round(np.sqrt(nb.shape[1]))),
        "mean_present": float(full.mean()),
        "frac_all_present": float((full == nb.shape[1]).mean()),
        "hist_present": {int(k): int(v) for k, v in zip(*np.unique(full, return_counts=True))},
    }


def masked_mean(X: np.ndarray, nb: np.ndarray) -> np.ndarray:
    out = np.zeros((len(nb), X.shape[1]), dtype=np.float32)
    cnt = np.zeros((len(nb), 1), dtype=np.float32)
    for s in range(nb.shape[1]):
        ids = nb[:, s]
        ok = ids >= 0
        out[ok] += X[ids[ok]]
        cnt[ok] += 1
    return out / np.maximum(cnt, 1)


def broken_pair_mean(X: np.ndarray, nb: np.ndarray, ortho: np.ndarray, seed: int) -> np.ndarray:
    """Moyenne de 9 voisins tirés au hasard dans le MÊME ortho (appariement rompu)."""
    rng = np.random.RandomState(seed)
    by_ortho = {int(o): np.where(ortho == o)[0] for o in np.unique(ortho)}
    out = np.zeros((len(nb), X.shape[1]), dtype=np.float32)
    picks = np.empty((len(nb), nb.shape[1]), dtype=np.int64)
    for i in range(len(nb)):
        cand = by_ortho[int(ortho[i])]
        picks[i] = rng.choice(cand, size=nb.shape[1], replace=True)
    for s in range(nb.shape[1]):
        out += X[picks[:, s]]
    return out / nb.shape[1]


# ----------------------------------------------------------------------------- sonde canonique
_G: dict = {}


def _probe_fit(task):
    """Fit unique (bras, C) — s'exécute dans un processus fils (fork, COW)."""
    arm, C = task
    t0 = time.time()
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, _G["repo"])
    from src.utils import make_canonical_lr
    Xtr, ytr = _G["Xtr"][arm], _G["ytr"]
    Xva, yva = _G["Xva"][arm], _G["yva"]
    sc = StandardScaler().fit(Xtr)
    lr = make_canonical_lr(C)
    lr.fit(sc.transform(Xtr), ytr)
    pred = lr.predict(sc.transform(Xva))
    return arm, C, macro_f1(yva, pred), round(time.time() - t0, 1)


def _probe_final(task):
    arm = task
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, _G["repo"])
    from src.utils import make_canonical_lr
    sc = StandardScaler().fit(_G["Xtr"][arm])
    lr = make_canonical_lr(_G["best_C"][arm])
    lr.fit(sc.transform(_G["Xtr"][arm]), _G["ytr"])
    pred = lr.predict(sc.transform(_G["Xte"][arm]))
    return arm, pred


def run_probes(arm_mats: dict, ytr, yva, yte, best_C_out, n_jobs=8, log=print) -> dict:
    """Sélection de C sur val + refit train complet + prédiction test, par bras."""
    repo = _G["repo"]
    out_dir = _G["out_dir"]
    _G.clear()
    _G.update({"Xtr": {a: m[0] for a, m in arm_mats.items()},
               "Xva": {a: m[1] for a, m in arm_mats.items()},
               "Xte": {a: m[2] for a, m in arm_mats.items()},
               "ytr": ytr, "yva": yva, "repo": repo, "out_dir": out_dir})
    tasks = [(a, C) for a in arm_mats for C in C_GRID]
    scores = {}
    ctx = get_context("fork")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        for arm, C, f1, dt in ex.map(_probe_fit, tasks):
            scores.setdefault(arm, {})[C] = f1
            log(f"[probe] {arm} C={C:g} valF1={f1:.4f} ({dt}s)")
    best = {}
    for arm, sc in scores.items():
        #égalité exacte -> plus grand C
        best[arm] = max(sc, key=lambda c: (round(sc[c], 12), c))
        best_C_out[arm] = {"best_C": best[arm], "val_scores": {f"{c:g}": sc[c] for c in C_GRID},
                           "val_margin": sc[best[arm]] - sorted(sc.values())[-2]
                           if len(sc) > 1 else 0.0}
        log(f"[probe] {arm}: best_C={best[arm]:g} valF1={sc[best[arm]]:.4f}")
    # sauvegarde intermédiaire : un timeout pendant les refits ne perd pas la sélection de C
    with open(os.path.join(_G["out_dir"], "probe_c_selection.json"), "w") as f:
        json.dump(best_C_out, f, indent=1)
    _G["best_C"] = best
    preds = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        for arm, pred in ex.map(_probe_final, list(arm_mats)):
            preds[arm] = pred
            np.save(os.path.join(_G["out_dir"], "preds", f"{arm}.npy"),
                    pred.astype(np.uint8))
    return preds


# ----------------------------------------------------------------------------- modèles appris
def build_torch_models(repo: str):
    import torch
    import torch.nn as nn

    class AttnAgg(nn.Module):
        """1 bloc d'attention sur k*k tokens + encodage positionnel 2D appris."""

        def __init__(self, d_in=768, d=128, heads=4, k=3, p=0.1):
            super().__init__()
            self.k = k
            self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
            self.pos = nn.Parameter(torch.zeros(1, k * k, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
            self.ln1 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p),
                                    nn.Linear(2 * d, d))
            self.ln2 = nn.LayerNorm(d)
            self.head = nn.Linear(d, N_CLASSES)

        def forward(self, tok, mask):           # tok (B,S,768), mask (B,S) bool = présent
            h = self.proj(tok) + self.pos[:, : tok.shape[1]]
            a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = self.ln1(h + a)
            h = self.ln2(h + self.ff(h))
            m = mask.unsqueeze(-1).float()
            h = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(h)

    class CNNAgg(nn.Module):
        """CNN 3×3 sur la grille de tokens (contrôle : même information, biais convolutionnel)."""

        def __init__(self, d_in=768, k=3, c1=64, c2=112, p=0.1):
            super().__init__()
            self.k = k
            self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, c1))
            self.net = nn.Sequential(
                nn.Conv2d(c1, c2, 3, padding=1), nn.GELU(), nn.Dropout2d(p),
                nn.Conv2d(c2, c2, 3, padding=1), nn.GELU(),
            )
            self.head = nn.Linear(c2, N_CLASSES)

        def forward(self, tok, mask):           # tok (B,k*k,768)
            b, s, _ = tok.shape
            h = self.proj(tok).transpose(1, 2).reshape(b, -1, self.k, self.k)
            h = self.net(h)
            h = h.mean(dim=(2, 3))
            return self.head(h)

    return AttnAgg, CNNAgg


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_learned(arm, k, feat, nb, y, split, seeds, out_dir, repo, log=print,
                  per_sample_perm=False, class_weighted=False, max_epochs=60, patience=10,
                  batch=512, lr=1e-3, wd=0.05, fixed_epochs=0):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    AttnAgg, CNNAgg = build_torch_models(repo)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat_t = torch.from_numpy(feat).to(dev)
    nb_t = torch.from_numpy(nb.astype(np.int64)).to(dev)
    y_t = torch.from_numpy(y.astype(np.int64)).to(dev)
    tr = np.where(split == 0)[0]
    va = np.where(split == 1)[0]
    te = np.where(split == 2)[0]
    # permutation par échantillon (contrôle de disposition) : fixée, indépendante du contenu
    perm = None
    if per_sample_perm:
        rng = np.random.RandomState(1234 + k)
        pm = np.stack([rng.permutation(k * k) for _ in range(len(nb))]).astype(np.int64)
        perm = torch.from_numpy(pm).to(dev)

    def gather(ids):
        g = nb_t[ids]
        if perm is not None:
            g = torch.gather(g, 1, perm[ids])
        m = g >= 0
        tok = feat_t[g.clamp(min=0)] * m.unsqueeze(-1)   # voisin absent -> vecteur nul
        return tok, m

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = (CNNAgg(k=k) if arm.startswith("L2") else AttnAgg(k=k)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        cw = None
        if class_weighted:
            cnt = np.bincount(y[tr], minlength=N_CLASSES).astype(np.float64)
            cw = torch.tensor((cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))),
                              dtype=torch.float32, device=dev)
        best = {"val_f1": -1.0, "epoch": -1, "state": None}
        rng = np.random.RandomState(seed)
        n_ep = fixed_epochs if fixed_epochs > 0 else max_epochs
        for ep in range(n_ep):
            model.train()
            order = rng.permutation(tr)
            for b in range(0, len(order), batch):
                ids = torch.from_numpy(order[b:b + batch]).to(dev)
                tok, m = gather(ids)
                logits = model(tok, m)
                loss = F.cross_entropy(logits, y_t[ids], weight=cw)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            model.eval()
            vp = []
            with torch.no_grad():
                for b in range(0, len(va), 1024):
                    ids = torch.from_numpy(va[b:b + 1024]).to(dev)
                    tok, m = gather(ids)
                    vp.append(model(tok, m).argmax(1).cpu().numpy())
            vf1 = macro_f1(y[va], np.concatenate(vp), labels=np.array(sorted(set(y[va].tolist()))))
            if ep % 5 == 0 or ep == n_ep - 1:
                log(f"[learned] {arm} seed={seed} ep={ep} valF1={vf1:.4f}")
            if vf1 > best["val_f1"] + 1e-9:
                best = {"val_f1": vf1, "epoch": ep,
                        "state": {kk: v.detach().clone() for kk, v in model.state_dict().items()}}
            if fixed_epochs > 0:
                continue                      # pas d'arrêt précoce : dernier checkpoint
            if ep - best["epoch"] >= patience:
                break
        if fixed_epochs == 0:
            model.load_state_dict(best["state"])
        model.eval()
        tp = []
        with torch.no_grad():
            for b in range(0, len(te), 1024):
                ids = torch.from_numpy(te[b:b + 1024]).to(dev)
                tok, m = gather(ids)
                tp.append(model(tok, m).argmax(1).cpu().numpy())
        pred = np.concatenate(tp).astype(np.int64)
        met = all_metrics(y[te], pred)
        log(f"[learned] {arm} seed={seed} ep={best['epoch']} valF1={best['val_f1']:.4f} "
            f"testF1={met['f1_macro_pres']:.4f}")
        results.append({"seed": seed, "epoch": best["epoch"], "val_f1": best["val_f1"],
                        "metrics": met, "pred": pred,
                        "n_params": n_params(model)})
    return results


# ----------------------------------------------------------------------------- bootstrap
def hierarchical_bootstrap(y, ortho, preds: dict, n_boot=10000, seed=7, log=print):
    """Bootstrap à 2 niveaux : orthos avec remise, puis tuiles avec remise dans chaque ortho.

    ``preds[a]`` peut être (n,) ou (S, n) : dans le second cas la statistique répliquée est la
    **moyenne inter-seed** des F1 (estimateur canonique du projet) — jamais un vote majoritaire.
    """
    rng = np.random.RandomState(seed)
    orthos = np.unique(ortho)
    idx_by_o = {int(o): np.where(ortho == o)[0] for o in orthos}
    arms = list(preds)
    preds = {k: (np.asarray(v) if np.asarray(v).ndim == 2 else np.asarray(v)[None, :])
             for k, v in preds.items()}
    boot = {a: np.zeros(n_boot, dtype=np.float32) for a in arms}
    labels = np.arange(N_CLASSES)
    for b in range(n_boot):
        chosen = rng.choice(orthos, size=len(orthos), replace=True)
        parts = [idx_by_o[int(o)][rng.randint(0, len(idx_by_o[int(o)]), len(idx_by_o[int(o)]))]
                 for o in chosen]
        ids = np.concatenate(parts)
        yt = y[ids]
        for a in arms:
            boot[a][b] = float(np.mean([macro_f1(yt, ps[ids], labels=labels)
                                        for ps in preds[a]]))
        if (b + 1) % 2000 == 0:
            log(f"[boot] {b + 1}/{n_boot}")
    out = {}
    for a in arms:
        lo, hi = np.percentile(boot[a], [2.5, 97.5])
        out[a] = {"mean": float(boot[a].mean()), "ci95": [float(lo), float(hi)]}
    return out, boot


def paired_delta(boot: dict, a: str, b: str) -> dict:
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_two_sided_boot": float(min((d <= 0).mean(), (d >= 0).mean()) * 2)}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--split", default=os.path.join(root, "out/pilot_split.json"))
    ap.add_argument("--features-dir", default=os.path.join(root, "out/features"))
    ap.add_argument("--out-dir", default=os.path.join(root, "out"))
    ap.add_argument("--repo", default=os.path.expanduser("~/benchmark-memoire"))
    ap.add_argument("--arms", default="R0,R1,R2,R3,R4,L1,L2,L1p")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--perm-reps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--fixed-epochs", type=int, default=0,
                    help=">0 : entraîne exactement N époques sans sélection sur validation")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "arms"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
    _G["repo"] = a.repo
    _G["out_dir"] = a.out_dir
    logf = open(os.path.join(a.out_dir, f"run_{a.tag}.log"), "a")

    def log(*m):
        s = " ".join(str(x) for x in m)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    log(f"[run] arms={arms} seeds={seeds} boot={a.boot}")

    d = load_all(a.split, a.features_dir)
    y, split, ortho = d["y"], d["split"], d["ortho"]
    Xt, Xc = d["X"]["tile"], d["X"]["ctx512"]
    log(f"[run] n={d['n']} train={int((split == 0).sum())} val={int((split == 1).sum())} "
        f"test={int((split == 2).sum())} dim={Xt.shape[1]}")

    nb3 = build_neighbors(ortho, d["row"], d["col"], 3)
    stats = {"k3": grid_stats(nb3)}
    if any(x in arms for x in ("L3", "L4", "R5", "R7")):
        for k in (5, 7):
            stats[f"k{k}"] = grid_stats(build_neighbors(ortho, d["row"], d["col"], k))
    json.dump(stats, open(os.path.join(a.out_dir, "grid_stats.json"), "w"), indent=1)
    log(f"[grid] {stats['k3']}")

    tr, va, te = (split == 0), (split == 1), (split == 2)
    probe_mats, probe_specs = {}, {}

    def add_probe(name, X):
        probe_mats[name] = (X[tr], X[va], X[te])
        probe_specs[name] = {"kind": "linear_probe", "dim": int(X.shape[1]),
                             "input": name}

    if "R0" in arms:
        add_probe("R0", Xt)
    if "R1" in arms:
        add_probe("R1", Xc)
    if "R2" in arms:
        add_probe("R2", np.concatenate([Xt, Xc], axis=1))
    if "R3" in arms:
        add_probe("R3", masked_mean(Xt, nb3))
    if "R4" in arms:
        for p in range(a.perm_reps):
            add_probe(f"R4p{p}", broken_pair_mean(Xt, nb3, ortho, seed=1000 + p))

    probe_preds, best_C = {}, {}
    if probe_mats:
        t0 = time.time()
        probe_preds = run_probes(probe_mats, y[tr], y[va], y[te], best_C,
                                 n_jobs=a.jobs, log=log)
        log(f"[probe] terminé en {time.time() - t0:.0f}s")

    # --- agrégats R4 (moyenne des 5 réplicats) : rapportée mais jamais utilisée pour un vote
    records = {}
    for name, pred in probe_preds.items():
        met = all_metrics(y[te], pred)
        np.save(os.path.join(a.out_dir, "preds", f"{name}.npy"), pred.astype(np.uint8))
        rec = {"arm": name, "spec": probe_specs[name], "metrics": met,
               "selection": best_C.get(name), "seeds": [None], "tag": a.tag}
        json.dump(rec, open(os.path.join(a.out_dir, "arms", f"{name}.json"), "w"), indent=1)
        records[name] = rec
        log(f"[arm] {name} testF1={met['f1_macro_pres']:.4f}")

    # --- bras appris
    learned_preds = {}
    if any(x in arms for x in ("L1", "L2", "L1p", "L3", "L4", "L1b")):
        LSPEC = {"L1": (3, False, False), "L2": (3, False, False), "L1p": (3, True, False),
                 "L3": (5, False, False), "L4": (7, False, False), "L1b": (3, False, True)}
        for name in [x for x in arms if x in LSPEC]:
            k, pperm, cw = LSPEC[name]
            nbk = nb3 if k == 3 else build_neighbors(ortho, d["row"], d["col"], k)
            t0 = time.time()
            res = train_learned(name, k, Xt, nbk, y, split, seeds, a.out_dir, a.repo,
                                log=log, per_sample_perm=pperm, class_weighted=cw,
                                max_epochs=a.epochs, patience=a.patience,
                                fixed_epochs=a.fixed_epochs)
            preds = np.stack([r["pred"] for r in res])
            met = {m: float(np.mean([r["metrics"][m] for r in res]))
                   for m in ("f1_macro_pres", "f1_macro_all", "accuracy",
                             "balanced_accuracy")}
            met["f1_macro_pres_std"] = float(np.std(
                [r["metrics"]["f1_macro_pres"] for r in res], ddof=1)) if len(res) > 1 else 0.0
            met["per_class_f1"] = {
                c: float(np.mean([r["metrics"]["per_class_f1"][c] for r in res]))
                for c in CLASS_NAMES_11}
            np.save(os.path.join(a.out_dir, "preds", f"{name}.npy"), preds.astype(np.uint8))
            rec = {"arm": name, "spec": {"kind": "learned", "k": k, "dim": 768,
                                         "per_sample_perm": pperm,
                                         "class_weighted": cw,
                                         "n_params": res[0]["n_params"]},
                   "metrics": met, "seeds": seeds,
                   "per_seed": [{k2: v for k2, v in r.items() if k2 != "pred"} for r in res],
                   "tag": a.tag, "elapsed_s": round(time.time() - t0, 1)}
            json.dump(rec, open(os.path.join(a.out_dir, "arms", f"{name}.json"), "w"),
                      indent=1)
            records[name] = rec
            learned_preds[name] = preds          # (S, n) : moyenne inter-seed dans le bootstrap
            log(f"[arm] {name} testF1={met['f1_macro_pres']:.4f}±{met['f1_macro_pres_std']:.4f}")

    # --- bootstrap hiérarchique (sur les prédictions seed-moyennées des bras appris)
    all_preds = {**{n: p for n, p in probe_preds.items()}, **learned_preds}
    if all_preds:
        boot_ci, boot = hierarchical_bootstrap(y[te], ortho[te], all_preds,
                                               n_boot=a.boot, log=log)
        np.savez_compressed(os.path.join(a.out_dir, "bootstrap_raw.npz"),
                           **{k2: v for k2, v in boot.items()})
        pairs = [("L1", "R2"), ("L1", "R1"), ("L1", "L2"), ("L1", "L1p")]
        deltas = {f"{x}-{z}": paired_delta(boot, x, z) for x, z in pairs
                  if x in boot and z in boot}
        n_test_ortho = len(set(ortho[te].tolist()))
        json.dump({"ci": boot_ci, "paired": deltas, "n_boot": a.boot,
                   "unit": f"orthomosaic ({n_test_ortho} test) then tiles", "seed": 7},
                  open(os.path.join(a.out_dir, "bootstrap.json"), "w"), indent=1)
        log("[boot] " + json.dumps(deltas))

    json.dump({"records": records, "n": d["n"], "split": os.path.abspath(a.split),
               "features_dir": os.path.abspath(a.features_dir)},
              open(os.path.join(a.out_dir, f"summary_{a.tag}.json"), "w"), indent=1)
    log("[run] terminé")


if __name__ == "__main__":
    main()
