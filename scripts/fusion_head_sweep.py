#!/usr/bin/env python3
"""Sweep de têtes de fusion NON LINÉAIRES sur les embeddings fusionnés [tuile;contexte].

POURQUOI
--------
Les probes canoniques sont LINÉAIRES : sur `[t; c]` elles calculent `w_t·t + w_c·c`,
une somme additivesans termes croisés — incapable d'exprimer « cette tuile est MOSS
seulement si son voisinage est SEDG ». Le `mlp_probe.py` de juillet (résultat : non-
linéarité ≈ +0,003 sur tuile seule) n'a porté QUE sur des vues uniques : le fused est
le seul endroit du dépôt où l'hypothèse « patterns plus compliqués » n'est pas encore
réfutée. Preuve que le gap existe : gelé-fusionné-sondé ≈ 0,495 vs fusion APPRISE
(R2) = 0,508 — ces +0,013 sont la part d'interaction que la linéarité laisse sur
la table.

CE QUE TESTE CE SCRIPT (par dossier d'embeddings fused, 2 vues de D dims chacune)
---------------------------------------------------------------------------------
  lin_lbfgs  : reproduction de la sonde canonique (baseline obligatoire ; mono-thread)
  lin_adamw  : contrôle de protocole — linaire entraîné par la même boucle AdamW que
               les têtes non linéaires (isolement du coût optimiseur, cf. mlp_probe)
  mlp2       : MLP 2 couches cachées (H, dropout, wd — grille petite)
  bil_diag   : bilinéaire diagonal  [Wt·t ; Wc·c ; (Wt·t)⊙(Wc·c)] → linéaire  → 11
               (termes croisés t_i·c_i alignés — le « contraste local »)
  film       : gamma,beta = Lin(c) ; z = (1+gamma)⊙(W t) + beta → linéaire → 11
               (le contexte module les canaux de la tuile)

Protocole (miroir de mlp_probe.py / src/probe.py) :
  - StandardScaler fit sur TRAIN, appliqué à val/test ;
  - sélection de config sur VAL (f1_macro_pres), moyenne des seeds de training ;
    le test ne participe jamais à la sélection ;
  - métrique rapportée : f1_macro_pres TEST, mean±std sur 3 seeds de training ;
  - delta citation = tete_nonlin - lin_adamw (à protocole égal), JAMAIS vs lbfgs
    (l'écart lbfgs↔adamw est un artefact d'optimiseur, documenté en juillet).
  - sur les dossiers 768d (vue seule : design A), seules lin_lbfgs/lin_adamw/mlp2
    tournent ; bil_diag/film exigent la fusion et sont sautés (signal `skipped`).

ENTRÉES (sur Narval) : dossiers `$SCRATCH/context_distill/sig_embeddings/<tag>/`
avec train/val/test.npy (+_labels.npy, 11 cls). Balayage par --tag-glob, défaut :
    *_FROZEN_fused_ctx*            (sweep gelé 5 backbones × 3 tailles, seed 0)
    *_ctxdistill_dB_tL_*           (R2 entraîné, 3 seeds — fusion apprise 1536)
    *_ctxdistill_dA_tL_* / _tEMA_* (design A, tuile seule — sanity : ≈ baseline)

SORTIES : <out-dir>/fusion_heads/<tag>.json  (idempotent : skip si existe)
          <out-dir>/fusion_heads/_aggregate.csv (régénéré à chaque run)

Usage Narval (via scripts/slurm_fusion_head_sweep.sh, ne pas lancer à la main) :
    python3 scripts/fusion_head_sweep.py --sig-dir $SCRATCH/context_distill/sig_embeddings
Usage local (test / tags rapatriés) :
    python3 scripts/fusion_head_sweep.py --sig-dir results/context_distill/sig_embeddings \
        --tag-glob '*_ctxdistill_dB_tL_*' --quick
"""
from __future__ import annotations

import os as _os
# MONO-THREAD avant tout import numpy/sklearn (AGENTS.md §4.8) : la baseline
# lin_lbfgs doit reproduire les chiffres canoniques.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import argparse
import csv
import fnmatch
import glob
import json
import sys
import time

import numpy as np

_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.dirname(_HERE))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.utils import make_canonical_lr
from src.metrics import eval_classifier  # f1_macro_pres / all


def _pres_f1(y_true, y_pred) -> float:
    return eval_classifier(np.asarray(y_true), np.asarray(y_pred),
                           n_classes=11)["f1_macro_pres"]


C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
TRAIN_SEEDS = (0, 1, 2)
# Grille volontairement petite : tenue en une nuit CPU (~25 tags × 4 têtes ×
# 8 configs × 3 seeds). H borné à 512 : au-delà, on sur-apprend 49k tuiles·11 cls.
HP_MLP = [(H, p, wd) for H in (256, 512) for p in (0.2, 0.5) for wd in (0.0, 1e-4)]


# ─────────────────────────── têtes torch ────────────────────────────
def _make_model(kind, x_dim, d_view, n_cls, hp):
    """d_view = dim d'UNE vue (None si x_dim non scindable)."""
    import torch
    import torch.nn as nn
    H, p, wd = hp
    if kind == "lin_adamw":
        return nn.Linear(x_dim, n_cls), wd

    class MLP2(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x_dim, H), nn.ReLU(), nn.Dropout(p),
                nn.Linear(H, H), nn.ReLU(), nn.Dropout(p),
                nn.Linear(H, n_cls))

        def forward(self, x):
            return self.net(x)

    class BilDiag(nn.Module):
        """[Wt t ; Wc c ; (Wt t) ⊙ (Wc c)] -> H -> logits. Le produit terme à
        terme après projections aligne les directions utiles des deux vues."""
        def __init__(self):
            super().__init__()
            self.wt = nn.Linear(d_view, H)   # opère sur la partie tuile seule
            self.wc = nn.Linear(d_view, H)
            self.head = nn.Linear(3 * H, n_cls)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            t, c = x[:, :d_view], x[:, d_view:]
            a, b = self.wt(t), self.wc(c)
            return self.head(self.drop(torch.cat([a, b, a * b], dim=1)))

    class FiLM(nn.Module):
        """gamma, beta générés par le contexte, appliqués à un linéaire de la tuile."""
        def __init__(self):
            super().__init__()
            self.wt = nn.Linear(d_view, H)
            self.gam = nn.Linear(d_view, H)
            self.beta = nn.Linear(d_view, H)
            self.head = nn.Linear(H, n_cls)
            self.drop = nn.Dropout(p)

        def forward(self, x):
            t, c = x[:, :d_view], x[:, d_view:]
            z = (1.0 + self.gam(c)) * self.wt(t) + self.beta(c)
            return self.head(self.drop(z))

    if kind == "mlp2":
        return MLP2(), wd
    if kind == "bil_diag":
        return BilDiag(), wd
    if kind == "film":
        return FiLM(), wd
    raise ValueError(kind)


def _run_head(kind, feats, hp, seed, epochs, x_dim, d_view, n_cls):
    import torch
    torch.manual_seed(seed)
    xtr, ytr = feats["train"]
    xva, yva = feats["val"]
    xte, yte = feats["test"]
    model, wd = _make_model(kind, x_dim, d_view, n_cls, hp)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xv = torch.tensor(xva, dtype=torch.float32)
    best_val, best_state, bad = -1.0, None, 0
    n = xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xv).argmax(1).numpy()
        f1v = _pres_f1(yva, pv)
        if f1v > best_val + 1e-5:
            best_val = f1v
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(torch.tensor(xte, dtype=torch.float32)).argmax(1).numpy()
    return _pres_f1(yte, pt), best_val, ep + 1


def load_feats(sig_dir: str, scaler_fit: bool = True):
    feats = {}
    for split in ("train", "val", "test"):
        X = np.load(_os.path.join(sig_dir, f"{split}.npy")).astype(np.float32)
        y = np.load(_os.path.join(sig_dir, f"{split}_labels.npy")).astype(np.int64)
        feats[split] = (X, y)
    sc = StandardScaler().fit(feats["train"][0])
    for split in feats:
        X, y = feats[split]
        feats[split] = (sc.transform(X), y)
    return feats


def probe_lbfgs(feats):
    """Sonde canonique : grille C, sélection val f1_macro_pres, test rapporté."""
    xtr, ytr = feats["train"]
    xva, yva = feats["val"]
    xte, yte = feats["test"]
    best = (-1, None)
    for C in C_GRID:
        lr = make_canonical_lr(C, max_iter=2000)
        lr.fit(xtr, ytr)
        f1v = _pres_f1(yva, lr.predict(xva))
        if f1v > best[0]:
            best = (f1v, C)
    _, C = best
    lr = make_canonical_lr(C, max_iter=2000)
    lr.fit(xtr, ytr)
    return {"f1_test": _pres_f1(yte, lr.predict(xte)), "f1_val": best[0], "best_C": C}


def sweep_tag(sig_dir, tag, quick=False):
    feats = load_feats(sig_dir)
    x_dim = feats["train"][0].shape[1]
    d_view = x_dim // 2 if FUSED_TAGS_RE.search(tag) else None
    epochs = 20 if quick else 120
    grid_mlp = HP_MLP[:2] if quick else HP_MLP
    out = {"tag": tag, "x_dim": x_dim, "fused": bool(d_view), "n_train": int(feats["train"][0].shape[0])}
    t0 = time.time()
    out["lin_lbfgs"] = probe_lbfgs(feats)
    kinds = ["lin_adamw", "mlp2"] + (["bil_diag", "film"] if d_view else [])
    for kind in kinds:
        grid = grid_mlp if kind != "lin_adamw" else [(0, 0.0, 0.0), (0, 0.0, 1e-4)]
        # sélection config sur val, moyennes test sur les seeds
        per_cfg = {}
        for hp in grid:
            f1s, f1vs = [], []
            for s in TRAIN_SEEDS:
                f1t, f1v, ep = _run_head(kind, feats, hp, s, epochs,
                                         x_dim, d_view, n_cls=11)
                f1s.append(f1t); f1vs.append(f1v)
            per_cfg[repr(hp)] = {"val_mean": float(np.mean(f1vs)), "test": f1s,
                                 "seeds": list(TRAIN_SEEDS)}
        best_cfg = max(per_cfg.items(), key=lambda kv: kv[1]["val_mean"])
        te = np.array(best_cfg[1]["test"])
        out[kind] = {"best_hp": best_cfg[0], "f1_mean": float(te.mean()),
                     "f1_std": float(te.std(ddof=1)), "test_per_seed": te.tolist(),
                     "delta_vs_lin_adamw": None,
                     "val_mean": best_cfg[1]["val_mean"]}
    if "lin_adamw" in out:
        for kind in kinds:
            if kind != "lin_adamw":
                out[kind]["delta_vs_lin_adamw"] = round(
                    out[kind]["f1_mean"] - out["lin_adamw"]["f1_mean"], 4)
    out["runtime_s"] = round(time.time() - t0, 1)
    return out


FUSED_TAGS_RE = None  # compilé dans main


def main() -> None:
    global FUSED_TAGS_RE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sig-dir", required=True, help="dossier parent des <tag>/{train,val,test}.npy")
    ap.add_argument("--out-dir", default=None, help="défaut : <sig-dir>/../fusion_heads")
    ap.add_argument("--tag-glob", nargs="*",
                    default=["*_FROZEN_fused_ctx*", "*_ctxdistill_dB_tL_*",
                             "*_ctxdistill_dA_tL_*", "*_ctxdistill_dA_tEMA_*"])
    ap.add_argument("--fused-tag-regex", default=r"(FROZEN_fused|ctxdistill_dB|_fused_)",
                    help="regex déterminant quels tags sont en fusion 2 vues")
    ap.add_argument("--only", nargs="*", default=None, help="restreint aux tags donnés")
    ap.add_argument("--quick", action="store_true", help="20 époques, 2 configs (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import re
    FUSED_TAGS_RE = re.compile(args.fused_tag_regex)

    tags = sorted({p.rstrip(_os.sep).split(_os.sep)[-1] for g in args.tag_glob
                   for p in glob.glob(_os.path.join(args.sig_dir, g))})
    if args.only:
        tags = [t for t in tags if any(fnmatch.fnmatch(t, p) for p in args.only)]
    out_dir = args.out_dir or _os.path.join(
        _os.path.dirname(args.sig_dir.rstrip("/")), "fusion_heads")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"[fusion-heads] {len(tags)} tags → {out_dir}", flush=True)
    for tag in tags:
        out_path = _os.path.join(out_dir, f"{tag}.json")
        if _os.path.exists(out_path) and not args.force:
            print(f"[fusion-heads] skip {tag} (existe)", flush=True)
            continue
        try:
            res = sweep_tag(_os.path.join(args.sig_dir, tag), tag, quick=args.quick)
        except Exception as e:  # un tag cassé ne tue pas la campagne
            print(f"[fusion-heads] ERROR {tag}: {e}", flush=True)
            continue
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        line = " | ".join(f"{k}:{v['f1_mean']:.4f}" for k, v in res.items()
                          if isinstance(v, dict) and "f1_mean" in v)
        lb = res.get("lin_lbfgs", {}).get("f1_test")
        print(f"[fusion-heads] OK {tag}  lbfgs={lb if lb is None else round(lb,4)}  {line}"
              f"  ({res['runtime_s']}s)", flush=True)
        # agrégat CSV régénéré
        rows = []
        for jp in sorted(glob.glob(_os.path.join(out_dir, "*.json"))):
            d = json.load(open(jp))
            for k in ("lin_lbfgs", "lin_adamw", "mlp2", "bil_diag", "film"):
                v = d.get(k)
                if not v:
                    continue
                rows.append({"tag": d["tag"], "fused": d["fused"], "head": k,
                             "f1_mean": round(v.get("f1_mean", v.get("f1_test")), 4),
                             "f1_std": round(v.get("f1_std", 0.0), 4),
                             "delta_vs_lin_adamw": v.get("delta_vs_lin_adamw", ""),
                             "best": v.get("best_hp", v.get("best_C", ""))})
        with open(_os.path.join(out_dir, "_aggregate.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
