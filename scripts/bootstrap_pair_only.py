"""Bootstrap ciblé pour la paire dinov3_vitb16_lvd vs vitb16_fulft_arctic.

Évite le chargement de tous les modèles pour rester dans le budget CPU.

Méthodologie canonique (alignée sur ``significance_matrix_all12.json``) :
- best_C LU depuis ``results/with_rhol/probe_knn_cgrid.json`` (sélection val,
  JAMAIS C forcé).
- bootstrap apparié n=1000, seed=42, métrique f1_macro_pres (classes présentes).

Usage: python scripts/bootstrap_pair_only.py
       python scripts/bootstrap_pair_only.py --n-bootstrap 10000  # pour le rapport
"""
import json
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import load_features
from src.probe import _standardize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from src.utils import make_canonical_lr

N_CLASSES = 12
SEED = 42
PAIR = ("dinov3_vitb16_lvd", "vitb16_fulft_arctic")
DEFAULT_PROBE = "results/with_rhol/probe_knn_cgrid.json"


def _load_best_C(probe_json: str) -> dict[str, float]:
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["best_C"]) for m, d in probe.items() if "best_C" in d}


def _refit_predict(cfg, model_key, C, seed=42):
    feats = load_features(cfg, model_key)
    ytr = np.asarray(feats["train"][1])
    yte = np.asarray(feats["test"][1])
    xtr, _xva, xte = _standardize(feats)
    clf = make_canonical_lr(
        C=C,
        max_iter=cfg.probe.max_iter,
        random_state=seed,
    )
    clf.fit(xtr, ytr)
    return yte, clf.predict(xte)


def _bootstrap(y_true, y_pred, n_boot, seed=42):
    n = len(y_true)
    rng = np.random.RandomState(seed)
    f1_pres = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        present = sorted({int(v) for v in yt})
        f1_pres[b] = f1_score(yt, yp, average="macro", zero_division=0, labels=present)
    return f1_pres


def main():
    ap_args = sys.argv[1:]
    n_boot = 1000
    probe_json = DEFAULT_PROBE
    # parse minimal : --n-bootstrap, --probe-json
    i = 0
    while i < len(ap_args):
        if ap_args[i] == "--n-bootstrap":
            n_boot = int(ap_args[i + 1]); i += 2
        elif ap_args[i] == "--probe-json":
            probe_json = ap_args[i + 1]; i += 2
        else:
            i += 1

    cfg = load_config("configs/frozen_eval.yaml")
    best_C = _load_best_C(probe_json)

    results = {}
    pres_samples = {}

    for model_key in PAIR:
        c = best_C.get(model_key)
        if c is None:
            raise SystemExit(
                f"[ERR] best_C absent pour {model_key} dans {probe_json}. "
                f"Lance d'abord probe.py --output-tag cgrid."
            )
        print(f"[{model_key}] fitting probe (best_C={c})...", flush=True)
        y_true, y_pred = _refit_predict(cfg, model_key, c, seed=SEED)
        obs_f1 = float(f1_score(
            y_true, y_pred, average="macro", zero_division=0,
            labels=sorted({int(v) for v in y_true}),
        ))
        print(f"[{model_key}] bootstrap {n_boot} iterations...", flush=True)
        samples = _bootstrap(y_true, y_pred, n_boot, seed=SEED)
        pres_samples[model_key] = samples

        results[model_key] = {
            "best_C": float(c),
            "n_test": int(len(y_true)),
            "f1_macro_pres": {
                "observed": float(obs_f1),
                "mean": float(samples.mean()),
                "std": float(samples.std(ddof=1)),
                "ci95_low": float(np.percentile(samples, 2.5)),
                "ci95_high": float(np.percentile(samples, 97.5)),
            },
        }

    # Pair comparison
    a, b = PAIR
    a_pres, b_pres = results[a]["f1_macro_pres"], results[b]["f1_macro_pres"]
    delta = a_pres["observed"] - b_pres["observed"]
    sa, sb = pres_samples[a], pres_samples[b]
    p_a_gt_b = float(np.mean(sa > sb))
    disjoint = (a_pres["ci95_low"] > b_pres["ci95_high"]) or \
               (b_pres["ci95_low"] > a_pres["ci95_high"])

    comparison = {
        "model_a": a,
        "model_b": b,
        "available": True,
        "best_C_a": float(best_C[a]),
        "best_C_b": float(best_C[b]),
        "delta_observed_a_minus_b": round(delta, 6),
        "p_a_gt_b": round(p_a_gt_b, 4),
        "ci95_a": [round(a_pres["ci95_low"], 6), round(a_pres["ci95_high"], 6)],
        "ci95_b": [round(b_pres["ci95_low"], 6), round(b_pres["ci95_high"], 6)],
        "ci95_disjoint": disjoint,
        "conclusion": "distinguishable" if disjoint else "not distinguishable",
        "note": ("Régénéré en best_C depuis probe_knn_cgrid.json (méthodo "
                 "canonique). L'ancienne version C=0.01 forcé est marquée "
                 ".deprecated."),
    }

    out = {
        "n_bootstrap": n_boot,
        "seed": SEED,
        "metric": "f1_macro_pres",
        "probe_source": probe_json,
        "models": {m: results[m] for m in PAIR},
        "pairs": [comparison],
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/bootstrap_pairs_controlled.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"PAIRE : {a} (gelé) vs {b} (fine-tuné)")
    print("=" * 60)
    print(f"  {a}  F1={a_pres['observed']:.4f}  IC95=[{a_pres['ci95_low']:.4f}, {a_pres['ci95_high']:.4f}]  C={best_C[a]}")
    print(f"  {b}  F1={b_pres['observed']:.4f}  IC95=[{b_pres['ci95_low']:.4f}, {b_pres['ci95_high']:.4f}]  C={best_C[b]}")
    print(f"  Δ (frozen - finetuned) = {delta:+.4f}")
    print(f"  P(frozen > finetuned)  = {p_a_gt_b:.3f}")
    print(f"  IC95 disjoints ? {disjoint}")
    print(f"  Conclusion : {comparison['conclusion']}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
