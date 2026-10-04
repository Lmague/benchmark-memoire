#!/usr/bin/env python3
"""Banc d'évaluation des modèles SSL Léo sur les annotations — kNN / logreg / MLP + ensemble.

CONTEXTE
--------
`annotations_leo/` contient plusieurs milliers de points annotés (5-6 espèces) sur l'ortho
Clairière, avec pour chacun l'embedding de la tuile 224 et du contexte 512. Jusqu'ici seul
le backbone SimDINOv2-B iNat **gelé** a été encodé (`embeddings/bank.npz`,
`negatives.npz`). `extract_embeddings.py --ckpt <pth> --out-dir embeddings/<tag>` produit
les mêmes fichiers pour n'importe quel checkpoint — dont les 3 seeds ExPLoRA-Léo.

Ce script compare, à protocole STRICTEMENT apparié (mêmes folds spatiaux et mêmes
négatifs pour tous les tags) :

  1. ``multiclass`` : classification des points annotés — logreg / kNN / MLP ;
  2. ``detection``  : un détecteur binaire par espèce (positifs vs {autres espèces + fond
     aléatoire + candidats rejetés}), métrique AUPRC comme `build_detectors.py` ;
  3. ``ensemble``   : moyenne des probabilités des seeds SSL, évaluée sur les MÊMES folds.

⚠️ STATUT DE L'ENSEMBLE (AGENTS.md §4.4)
   L'estimateur canonique du mémoire est ``mean±std`` sur les seeds ; le **vote
   majoritaire / la moyenne de probas n'est PAS un estimateur canonique** et ne doit pas
   être cité comme F1 de modèle dans le manuscrit. Ici l'ensemble sert (a) de diagnostic
   de variance et (b) de score pour l'outil d'annotation. Les deux chiffres sont
   reportés côte à côte, jamais fusionnés.

USAGE
    # 1) extraire les embeddings de chaque modèle (local, CPU) :
    #      bash scripts/leo_ssl_annot_extract.sh
    # 2) comparer :
    python scripts/leo_ssl_annot_bench.py --clf logreg,knn,mlp
    # 3) détecteurs pour l'outil d'annotation + scan de la grille :
    python scripts/leo_ssl_annot_bench.py --clf logreg --save-detectors --score-grid
"""
from __future__ import annotations

# --- Mono-thread BLAS AVANT numpy/sklearn (AGENTS.md §4.8) -----------------------
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np

SPECIES = ["Lotcorn", "Leuvul", "Ascsyr", "Daucar", "Solcan"]   # Eumac (n=1) exclu
BLOCK_M = 10.0
N_SPLITS = 5
DEFAULT_EMB = Path("/home/erazal/annotations_leo/embeddings")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def pick(tile: np.ndarray, ctx: np.ndarray, variant: str) -> np.ndarray:
    if variant == "tile":
        return tile
    if variant == "ctx":
        return ctx
    if variant == "fused":
        return np.hstack([tile, ctx])
    raise ValueError(f"variante inconnue : {variant}")


# ------------------------------------------------------------------ chargement
def discover_tags(emb_dir: Path, wanted: list[str] | None) -> list[str]:
    """``base`` = le bank.npz historique (racine) ou ``base/bank.npz`` ; les autres tags
    sont les sous-dossiers contenant un bank.npz."""
    tags = []
    if (emb_dir / "bank.npz").exists() or (emb_dir / "base" / "bank.npz").exists():
        tags.append("base")
    tags += sorted(d.name for d in emb_dir.iterdir()
                   if d.is_dir() and d.name != "base" and (d / "bank.npz").exists())
    if wanted:
        tags = [t for t in wanted if t in tags]
    return tags


def tag_dir(emb_dir: Path, tag: str) -> Path:
    if tag == "base" and not (emb_dir / "bank.npz").exists() \
            and (emb_dir / "base" / "bank.npz").exists():
        return emb_dir / "base"
    return emb_dir if tag == "base" else emb_dir / tag


def load_bank(emb_dir: Path, tag: str) -> dict:
    d = np.load(tag_dir(emb_dir, tag) / "bank.npz", allow_pickle=True)
    ctx = l2(d["emb_ctx"])
    # les embeddings extraits depuis les JPEG web n'ont PAS de contexte 512 (cf.
    # leo_ssl_annot_extract_web.py) : emb_ctx y est nul et meta.json porte ctx=false.
    has_ctx = ctx.size > 0 and float(np.abs(ctx).max()) > 0.0
    meta_p = tag_dir(emb_dir, tag) / "meta.json"
    if meta_p.exists():
        try:
            has_ctx = has_ctx and bool(json.loads(meta_p.read_text()).get("ctx", True))
        except Exception:  # noqa: BLE001 — meta absent/corrompu : on garde la détection par les zéros
            pass
    return {"species": d["species"].astype(str),
            "x": d["x"].astype(float), "y": d["y"].astype(float),
            "tile": l2(d["emb_tile"]), "ctx": ctx, "has_ctx": has_ctx,
            "tr": np.asarray(d["tr"]), "tc": np.asarray(d["tc"])}


def load_negatives(emb_dir: Path, tag: str) -> dict | None:
    p = tag_dir(emb_dir, tag) / "negatives.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {"bg_tile": l2(d["bg_tile"]), "bg_ctx": l2(d["bg_ctx"]),
            "bg_x": d["bg_x"].astype(float), "bg_y": d["bg_y"].astype(float),
            "rej_tile": l2(d["rej_tile"]), "rej_ctx": l2(d["rej_ctx"]),
            "rej_species": d["rej_species"].astype(str),
            "rej_x": d["rej_x"].astype(float), "rej_y": d["rej_y"].astype(float)}


def assert_aligned(banks: dict[str, dict]) -> None:
    """Mêmes annotations, même ordre pour tous les tags — sinon les folds ne sont pas appariés."""
    ref_tag = next(iter(banks))
    ref = banks[ref_tag]
    for t, b in banks.items():
        same = (len(b["species"]) == len(ref["species"])
                and np.array_equal(b["species"], ref["species"])
                and np.allclose(b["x"], ref["x"]) and np.allclose(b["y"], ref["y"]))
        if not same:
            raise SystemExit(
                f"[bench] '{t}' n'est pas aligné sur '{ref_tag}' (nombre/ordre des points). "
                "Ré-extraire ce tag avec le même session.gpkg (les annotations ont changé ?).")


def assert_same_negatives(negs: dict[str, dict], ref_tag: str) -> None:
    ref = negs[ref_tag]
    for t, n in negs.items():
        if n is None:
            continue
        if (len(n["bg_x"]) != len(ref["bg_x"]) or not np.allclose(n["bg_x"], ref["bg_x"])
                or not np.allclose(n["bg_y"], ref["bg_y"])
                or len(n["rej_x"]) != len(ref["rej_x"])
                or not np.allclose(n["rej_x"], ref["rej_x"])):
            raise SystemExit(f"[bench] négatifs de '{t}' ≠ '{ref_tag}' (tuiles non appariées).")


# ------------------------------------------------------------------ classifieurs
def make_classifier(name: str, **kw):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=kw.get("C", 1.0), max_iter=3000,
                                                    class_weight="balanced"))])
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=kw.get("k", 5), metric="cosine",
                                    weights="distance")
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("sc", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4,
                                               early_stopping=True, n_iter_no_change=10,
                                               max_iter=500, random_state=0))])
    raise ValueError(f"classifieur inconnu : {name}")


def proba_full(Xtr, ytr, Xte, classes: np.ndarray, clf_name: str, **kw) -> np.ndarray:
    """``predict_proba`` réaligné sur ``classes`` (toutes les colonnes toujours présentes)."""
    pipe = make_classifier(clf_name, **kw)
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)
    full = np.zeros((len(Xte), len(classes)), dtype=np.float64)
    pos = {int(c): j for j, c in enumerate(classes)}
    for j, c in enumerate(pipe.classes_):
        full[:, pos[int(c)]] = p[:, j]
    return full


def spatial_groups(x, y, block_m: float = None) -> np.ndarray:
    b = BLOCK_M if block_m is None else block_m
    return (np.floor(np.asarray(x) / b).astype(np.int64) * 1000000
            + np.floor(np.asarray(y) / b).astype(np.int64))


def make_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = None):
    from sklearn.model_selection import StratifiedGroupKFold
    n = N_SPLITS if n_splits is None else n_splits
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=0)
                .split(np.zeros((len(y), 1)), y, groups))


def oof_proba(X, y, folds, classes: np.ndarray, clf_name: str, **kw) -> np.ndarray:
    P = np.zeros((len(y), len(classes)))
    for trn, te in folds:
        if len(np.unique(y[trn])) < 2:
            continue
        P[te] = proba_full(X[trn], y[trn], X[te], classes, clf_name, **kw)
    return P


# ------------------------------------------------------------------ métriques
def metrics_multiclass(P: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    pred = classes[P.argmax(1)]
    rec = recall_score(y, pred, average=None, labels=list(classes), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", labels=list(classes),
                                       zero_division=0)),
            "recall_per_class": {SPECIES[int(c)]: float(r) for c, r in zip(classes, rec)}}


def metrics_detection(score: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score)
    m = ~np.isnan(score)
    if len(np.unique(y[m])) < 2 or y[m].sum() < 3:
        return {"n_pos": int(y[m].sum()), "AUPRC": None, "ROC_AUC": None,
                "best_F1": None, "best_thr": None}
    prec, rec, thr = precision_recall_curve(y[m], score[m])
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1))
    return {"n_pos": int(y[m].sum()), "n_neg": int((y[m] == 0).sum()),
            "AUPRC": float(average_precision_score(y[m], score[m])),
            "ROC_AUC": float(roc_auc_score(y[m], score[m])),
            "best_F1": float(f1[k]),
            "best_thr": float(thr[min(k, len(thr) - 1)])}


def agg_mean_std(values) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


# ------------------------------------------------------------------ multiclass
def run_multiclass(banks: dict[str, dict], tags: list[str], variants: list[str],
                   clf_name: str, ensemble_tags: list[str], **kw) -> dict:
    ref = banks[tags[0]]
    keep = np.isin(ref["species"], SPECIES)
    y = np.array([SPECIES.index(s) for s in ref["species"][keep]])
    classes = np.arange(len(SPECIES))
    folds = make_folds(y, spatial_groups(ref["x"][keep], ref["y"][keep]))
    out = {"clf": clf_name, "n_points": int(keep.sum()), "n_classes": len(SPECIES),
           "n_folds": len(folds), "variants": {}}
    for variant in variants:
        P = {t: oof_proba(pick(banks[t]["tile"][keep], banks[t]["ctx"][keep], variant),
                          y, folds, classes, clf_name, **kw) for t in tags}
        ens = [t for t in ensemble_tags if t in P]
        d = {"per_tag": {t: metrics_multiclass(P[t], y, classes) for t in tags},
             "canonical_mean_std_ssl_accuracy": agg_mean_std(
                 [d_["accuracy"] for t, d_ in
                  ((t, metrics_multiclass(P[t], y, classes)) for t in ens)]),
             "canonical_mean_std_ssl_f1_macro": agg_mean_std(
                 [metrics_multiclass(P[t], y, classes)["f1_macro"] for t in ens]),
             "ensemble_diagnostic": None}
        if ens:
            P_ens = np.mean([P[t] for t in ens], axis=0)
            d["ensemble_diagnostic"] = {"members": ens, **metrics_multiclass(P_ens, y, classes)}
        out["variants"][variant] = d
    return out


# ------------------------------------------------------------------ détection
def detection_xy(bank: dict, neg: dict, sp: str):
    """(X_pos, X_neg, coords) pour l'espèce ``sp`` — identique pour tous les tags."""
    sp_all = bank["species"]
    pos = sp_all == sp
    oth = ~pos
    blocks_x = [bank["x"][pos], bank["x"][oth]]
    blocks_y = [bank["y"][pos], bank["y"][oth]]
    sel = neg["rej_species"] == sp
    if sel.any():
        blocks_x.append(neg["rej_x"][sel]); blocks_y.append(neg["rej_y"][sel])
    blocks_x.append(neg["bg_x"]); blocks_y.append(neg["bg_y"])
    n = [int(pos.sum()), int(oth.sum())] + ([int(sel.sum())] if sel.any() else []) + [len(neg["bg_x"])]
    return pos, oth, sel, np.concatenate(blocks_x), np.concatenate(blocks_y), n


def detection_matrix(bank: dict, neg: dict, sp: str, variant: str):
    pos, oth, sel, xs, ys, n = detection_xy(bank, neg, sp)
    blocks = [pick(bank["tile"][pos], bank["ctx"][pos], variant),
              pick(bank["tile"][oth], bank["ctx"][oth], variant)]
    if sel.any():
        blocks.append(pick(neg["rej_tile"][sel], neg["rej_ctx"][sel], variant))
    blocks.append(pick(neg["bg_tile"], neg["bg_ctx"], variant))
    X = np.concatenate(blocks, axis=0).astype(np.float32)
    yv = np.r_[np.ones(n[0]), np.zeros(int(sum(n[1:])))].astype(int)
    return X, yv, xs, ys


def run_detection(banks: dict[str, dict], negs: dict[str, dict], tags: list[str],
                  variants: list[str], clf_name: str, ensemble_tags: list[str], **kw) -> dict:
    ref, neg_ref = banks[tags[0]], negs[tags[0]]
    sp_all = ref["species"]
    out = {"clf": clf_name, "variants": {}}
    for variant in variants:
        per_sp = {}
        for sp in SPECIES:
            if (sp_all == sp).sum() < 5:
                continue
            _, yv, xs, ys = detection_matrix(ref, neg_ref, sp, variant)
            folds = make_folds(yv, spatial_groups(xs, ys))
            classes = np.array([0, 1])
            scores = {}
            for t in tags:
                if negs.get(t) is None:
                    continue
                X, _, _, _ = detection_matrix(banks[t], negs[t], sp, variant)
                if len(X) != len(yv):
                    raise SystemExit(
                        f"[bench] {t} / {sp} / {variant} : matrice {X.shape} ≠ y {yv.shape} — "
                        "négatifs NON appariés entre tags (rejeter/ré-extraire avec le même "
                        "candidates.sqlite et les mêmes annotations).")
                scores[t] = oof_proba(X, yv, folds, classes, clf_name, **kw)[:, 1]
            ens = [t for t in ensemble_tags if t in scores]
            per_tag = {t: metrics_detection(scores[t], yv) for t in scores}
            d = {"n_pos": int(yv.sum()), "n_neg": int((yv == 0).sum()), "n_folds": len(folds),
                 "per_tag": per_tag,
                 "canonical_mean_std_ssl_AUPRC": agg_mean_std(
                     [per_tag[t]["AUPRC"] for t in ens]),
                 "ensemble_diagnostic": None}
            if ens:
                s_ens = np.mean([scores[t] for t in ens], axis=0)
                d["ensemble_diagnostic"] = {"members": ens, **metrics_detection(s_ens, yv)}
            per_sp[sp] = d
        out["variants"][variant] = per_sp
    return out


# ------------------------------------------------------------------ détecteurs / grille
def save_detectors(emb_dir: Path, tag: str, bank: dict, neg: dict, variant: str,
                   C: float = 1.0) -> Path:
    """Ré-ajuste les détecteurs logreg sur TOUTES les données ; format de
    ``detectors_final_v2.npz`` (mean/scale/coef/intercept par espèce) — rechargeable tel quel
    par l'outil d'annotation et par ``--score-grid``."""
    species, arrays = [], {}
    for sp in SPECIES:
        if (bank["species"] == sp).sum() < 5:
            continue
        X, yv, _, _ = detection_matrix(bank, neg, sp, variant)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(X), yv)
        species.append(sp)
        arrays[f"{sp}_mean"] = sc.mean_
        arrays[f"{sp}_scale"] = sc.scale_
        arrays[f"{sp}_coef"] = np.asarray(clf.coef_).ravel()
        arrays[f"{sp}_intercept"] = np.atleast_1d(clf.intercept_)
    arrays["species"] = np.array(species)
    arrays["variant"] = np.array([variant])
    arrays["l2"] = np.array([True])
    p = tag_dir(emb_dir, tag) / f"detectors_{variant}.npz"
    np.savez_compressed(p, **arrays)
    return p


def _grid_matrix(emb_dir: Path, tag: str, variant: str):
    """Charge la grille d'un tag pour une variante (tile / ctx / fused)."""
    d = tag_dir(emb_dir, tag)
    if variant == "tile":
        return np.load(d / "grid_tile.npy", mmap_mode="r")
    if variant == "ctx":
        return np.load(d / "grid_ctx.npy", mmap_mode="r")
    if variant == "fused":
        if not (d / "grid_tile.npy").exists() or not (d / "grid_ctx.npy").exists():
            raise FileNotFoundError(d / "grid_tile.npy")
        return np.hstack([np.asarray(np.load(d / "grid_tile.npy", mmap_mode="r"), dtype=np.float16),
                          np.asarray(np.load(d / "grid_ctx.npy", mmap_mode="r"), dtype=np.float16)])
    raise ValueError(variant)


def score_grid(emb_dir: Path, tags: list[str], variant: str, out_name: str) -> None:
    """Applique les détecteurs de chaque tag à SA grille, moyenne → scores d'ensemble."""
    dets, grids = {}, {}
    for t in tags:
        dp = tag_dir(emb_dir, t) / f"detectors_{variant}.npz"
        if not dp.exists():
            log(f"[grid] {t} : {dp.name} absent → exclu de l'ensemble")
            continue
        try:
            g = _grid_matrix(emb_dir, t, variant)
        except FileNotFoundError:
            log(f"[grid] {t} : grille {variant} absente → exclu de l'ensemble")
            continue
        dets[t] = np.load(dp, allow_pickle=True)
        grids[t] = g
    if not dets:
        log("[grid] aucun tag exploitable — lancer d'abord --save-detectors et la phase grid")
        return
    sp_ref = [str(s) for s in dets[next(iter(dets))]["species"]]
    n0 = grids[next(iter(grids))].shape[0]
    S = np.zeros((n0, len(sp_ref)))
    for t, d in dets.items():
        if grids[t].shape[0] != n0:
            raise SystemExit(f"[grid] {t} : grille {grids[t].shape} ≠ {n0} lignes")
        X = l2(np.asarray(grids[t], dtype=np.float32))
        for j, sp in enumerate(sp_ref):
            z = (X - d[f"{sp}_mean"]) / d[f"{sp}_scale"]
            S[:, j] += 1.0 / (1.0 + np.exp(-(z @ d[f"{sp}_coef"]
                                             + float(d[f"{sp}_intercept"][0]))))
    S /= len(dets)
    out = emb_dir / f"{out_name}.npy"
    np.save(out, S.astype(np.float32))
    (emb_dir / f"{out_name}.json").write_text(json.dumps(
        {"species": sp_ref, "members": sorted(dets), "variant": variant,
         "note": "moyenne des probas (AGENTS.md §4.4 : non canonique, outil d'annotation)"},
        indent=2))
    log(f"[grid] {out} {S.shape} (membres : {sorted(dets)})")


# ------------------------------------------------------------------ rapport
def _table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def write_report(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench_annot.json").write_text(json.dumps(result, indent=2, default=str))
    tags = result["tags"]
    L = ["# SSL Léo sur les annotations — comparaison appariée", "",
         f"Tags : `{'`, `'.join(tags)}` — ensemble : `{'`, `'.join(result['ensemble_tags'])}`",
         f"Folds spatiaux appariés : {result['protocol']['n_splits']} plis, blocs "
         f"{result['protocol']['blocks_m']} m", "",
         "> ⚠️ `mean±std` sur les seeds = estimateur canonique (AGENTS.md §4.4). L'ensemble "
         "(moyenne des probas) est un **diagnostic / outil d'annotation**, pas un chiffre de "
         "manuscrit.", ""]
    mc = result.get("multiclass") or {}
    for clf, per_var in mc.items():
        L += [f"## Multiclass — {clf}", ""]
        rows = []
        for variant, d in per_var["variants"].items():
            for t in tags:
                if t in d["per_tag"]:
                    rows.append([variant, t, f"{d['per_tag'][t]['accuracy']:.4f}",
                                 f"{d['per_tag'][t]['f1_macro']:.4f}"])
            e = d["ensemble_diagnostic"]
            if e:
                rows.append([variant, f"**ensemble ({len(e['members'])})**",
                             f"**{e['accuracy']:.4f}**", f"**{e['f1_macro']:.4f}**"])
            ms = d["canonical_mean_std_ssl_accuracy"]
            rows.append([variant, "mean±std seeds", f"{ms['mean']:.4f} ± {ms['std']:.4f}", "—"])
        L += _table(["variant", "modèle", "accuracy", "F1-macro"], rows) + [""]
    det = result.get("detection") or {}
    for clf, per_var in det.items():
        L += [f"## Détection par espèce (AUPRC) — {clf}", ""]
        rows = []
        for variant, per_sp in per_var["variants"].items():
            for sp, d in per_sp.items():
                for t in tags:
                    if t in d["per_tag"]:
                        rows.append([variant, sp, t, f"{d['per_tag'][t]['AUPRC']:.4f}"])
                e = d["ensemble_diagnostic"]
                if e:
                    rows.append([variant, sp, f"**ensemble ({len(e['members'])})**",
                                 f"**{e['AUPRC']:.4f}**"])
        L += _table(["variant", "espèce", "modèle", "AUPRC"], rows) + [""]
    (out_dir / "bench_annot.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main() -> None:
    global BLOCK_M, N_SPLITS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", default=str(DEFAULT_EMB))
    ap.add_argument("--tags", default=None, help="liste séparée par virgules (défaut : auto)")
    ap.add_argument("--ensemble", default=None,
                    help="tags de l'ensemble (défaut : tous sauf 'base')")
    ap.add_argument("--clf", default="logreg", help="logreg | knn | mlp (liste possible)")
    ap.add_argument("--variants", default="tile", help="tile | ctx | fused (liste possible)")
    ap.add_argument("--task", default="both", choices=["multiclass", "detection", "both"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5, help="k du kNN")
    ap.add_argument("--blocks", type=float, default=BLOCK_M)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-detectors", action="store_true")
    ap.add_argument("--score-grid", action="store_true")
    ap.add_argument("--grid-out", default="grid_scores_ensemble")
    args = ap.parse_args()

    BLOCK_M, N_SPLITS = args.blocks, args.n_splits

    emb_dir = Path(args.emb_dir)
    tags = discover_tags(emb_dir,
                         [t.strip() for t in args.tags.split(",")] if args.tags else None)
    if len(tags) < 2:
        raise SystemExit(f"[bench] {emb_dir} : tags trouvés = {tags}. Extraire d'abord les "
                         "autres modèles : extract_embeddings.py --ckpt <pth> --out-dir "
                         "embeddings/<tag> --phase bank,negatives")
    ens = ([t.strip() for t in args.ensemble.split(",")] if args.ensemble
           else [t for t in tags if t != "base"])
    variants = [v.strip() for v in args.variants.split(",")]
    clf_names = [c.strip() for c in args.clf.split(",")]
    log(f"tags={tags} | ensemble={ens} | variants={variants} | clf={clf_names}")

    banks = {t: load_bank(emb_dir, t) for t in tags}
    assert_aligned(banks)
    no_ctx = [t for t in tags if not banks[t]["has_ctx"]]
    if no_ctx:
        drop = [v for v in variants if v in ("ctx", "fused")]
        if drop:
            log(f"[bench] {no_ctx} sans contexte 512 (embeddings JPEG web) → "
                f"variantes retirées : {drop}")
            variants = [v for v in variants if v not in drop]
        if not variants:
            raise SystemExit("[bench] plus aucune variante à évaluer")
    negs = {t: load_negatives(emb_dir, t) for t in tags}
    if any(v is not None for v in negs.values()):
        ref_neg = next((t for t in tags if negs[t] is not None), None)
        assert_same_negatives({k: v for k, v in negs.items() if v is not None}, ref_neg)
    if args.task in ("detection", "both") and negs.get(tags[0]) is None:
        log("[bench] pas de negatives.npz → tâche détection ignorée")
        args.task = "multiclass"

    result = {"tags": tags, "ensemble_tags": ens, "variants": variants,
              "clf_names": clf_names, "emb_dir": str(emb_dir),
              "protocol": {"blocks_m": BLOCK_M, "n_splits": N_SPLITS, "folds_paired": True,
                           "embedding_norm": "L2 puis StandardScaler (logreg/mlp), cosine (knn)",
                           "canonical_estimator": "mean±std sur les seeds (AGENTS.md §4.4)",
                           "ensemble_status": "diagnostic / outil d'annotation — NON canonique",
                           "n_points_multiclass": int(np.isin(banks[tags[0]]["species"],
                                                              SPECIES).sum())},
              "multiclass": {}, "detection": {}}
    for clf in clf_names:
        t0 = time.time()
        kw = {"k": args.k} if clf == "knn" else {"C": args.C}
        if args.task in ("multiclass", "both"):
            result["multiclass"][clf] = run_multiclass(banks, tags, variants, clf, ens, **kw)
        if args.task in ("detection", "both"):
            result["detection"][clf] = run_detection(banks, negs, tags, variants, clf, ens, **kw)
        log(f"  {clf} : {time.time() - t0:.0f}s")

    out_dir = Path(args.out_dir) if args.out_dir else emb_dir
    write_report(out_dir, result)
    log(f"→ {out_dir / 'bench_annot.json'} + bench_annot.md")

    if args.save_detectors:
        if clf_names[0] != "logreg":
            raise SystemExit("[bench] --save-detectors n'est implémenté que pour logreg "
                             "(le format detectors_*.npz stocke coef/intercept linéaires)")
        for t in tags:
            if negs.get(t) is None:
                continue
            for v in variants:
                p = save_detectors(emb_dir, t, banks[t], negs[t], v, C=args.C)
            log(f"  détecteurs {t} → {p} ({len(variants)} variante(s))")
    if args.score_grid:
        # l'ensemble de la grille suit --ensemble (par défaut : les seeds SSL seules,
        # comme l'outil d'annotation d'origine — ajouter 'base' pour inclure le gelé)
        score_grid(emb_dir, ens, variants[0], args.grid_out)


if __name__ == "__main__":
    main()
