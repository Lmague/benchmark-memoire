#!/usr/bin/env python3
"""Correction pour comparaisons multiples sur les paires de significance_matrix.json.

``significance_matrix.json`` fournit, pour chaque paire (A, B), la proportion bootstrap
``p_a_gt_b = P(F1_A > F1_B)`` (test apparié, n=1000). Ce script en dérive un p bilatéral
puis applique deux corrections sur l'ensemble des m = N(N-1)/2 paires :

  - Holm-Bonferroni (contrôle du FWER, step-down) ;
  - Benjamini-Hochberg (contrôle du FDR, step-up).

p bilatéral (par paire) :  p2 = 2 * min(p_a_gt_b, 1 - p_a_gt_b),
plancher à 1/n_bootstrap côté unilatéral (donc p2 >= 2/n_bootstrap) car le bootstrap ne
résout pas en dessous de 1/n. Les paires à p_a_gt_b = 0 ou 1 sont donc reportées à
p2 = 2/n_bootstrap (borne supérieure : "p <= 2/n").

Ne nécessite AUCUN embedding : pur post-traitement de significance_matrix.json.

Usage :
  python scripts/multiple_comparison.py \\
      --input results/significance_matrix.json \\
      --output results/significance_corrected.json \\
      --alpha 0.05
"""
from __future__ import annotations

import argparse
import json
import os


def _two_sided_p(p_a_gt_b: float, n_boot: int) -> tuple[float, bool]:
    """p bilatéral à partir de la proportion unilatérale, avec plancher 1/n_boot.

    Retourne (p2, floored) où ``floored`` indique que la valeur a touché le plancher
    (p_a_gt_b ∈ {0, 1} → p réel < 1/n_boot, on reporte la borne 2/n_boot).
    """
    one = min(p_a_gt_b, 1.0 - p_a_gt_b)
    floored = one <= 0.0
    one = max(one, 1.0 / n_boot)
    return min(2.0 * one, 1.0), floored


def holm_adjusted(pvals: list[float]) -> list[float]:
    """p-values ajustées de Holm-Bonferroni (step-down), monotones, plafonnées à 1."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)            # monotonie
        adj[idx] = min(running, 1.0)
    return adj


def bh_adjusted(pvals: list[float]) -> list[float]:
    """p-values ajustées de Benjamini-Hochberg (q-values, step-up), monotones, ≤ 1."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):          # du plus grand p au plus petit
        idx = order[rank]
        val = pvals[idx] * m / (rank + 1)
        running = min(running, val)            # monotonie (enveloppe inférieure)
        adj[idx] = min(running, 1.0)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/significance_matrix.json")
    ap.add_argument("--output", default="results/significance_corrected.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    n_boot = int(data.get("n_bootstrap", 1000))
    pairs = data["pairs"]

    keys = list(pairs.keys())
    p2, floored = [], []
    for k in keys:
        p, fl = _two_sided_p(float(pairs[k]["p_a_gt_b"]), n_boot)
        p2.append(p)
        floored.append(fl)

    holm = holm_adjusted(p2)
    bh = bh_adjusted(p2)

    out_pairs = {}
    for i, k in enumerate(keys):
        v = pairs[k]
        out_pairs[k] = {
            "model_a": v["model_a"],
            "model_b": v["model_b"],
            "delta_observed_a_minus_b": v["delta_observed_a_minus_b"],
            "p_a_gt_b": v["p_a_gt_b"],
            "p_two_sided": p2[i],
            "p_two_sided_floored": floored[i],
            "p_holm": holm[i],
            "p_bh": bh[i],
            "reject_holm": bool(holm[i] < args.alpha),
            "reject_bh": bool(bh[i] < args.alpha),
            "ci95_disjoint": v.get("ci95_disjoint"),
        }

    n_raw = sum(p < args.alpha for p in p2)
    n_holm = sum(v["reject_holm"] for v in out_pairs.values())
    n_bh = sum(v["reject_bh"] for v in out_pairs.values())
    n_disj = sum(bool(v.get("ci95_disjoint")) for v in out_pairs.values())

    out = {
        "source": args.input,
        "metric": data.get("metric", "f1_macro_pres"),
        "n_bootstrap": n_boot,
        "alpha": args.alpha,
        "n_pairs": len(keys),
        "n_models": len(data.get("models", [])),
        "method_note": (
            "p bilatéral = 2*min(p_a_gt_b, 1-p_a_gt_b), plancher 1/n_bootstrap ; "
            "Holm (FWER) et BH (FDR) appliqués sur les N(N-1)/2 paires."
        ),
        "summary": {
            "significant_raw": n_raw,
            "significant_holm": n_holm,
            "significant_bh": n_bh,
            "ci95_disjoint": n_disj,
        },
        "pairs": out_pairs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[multiple_comparison] {len(keys)} paires, alpha={args.alpha}")
    print(f"  significatives brut (p<{args.alpha})      : {n_raw}/{len(keys)}")
    print(f"  significatives Holm (FWER)                : {n_holm}/{len(keys)}")
    print(f"  significatives BH   (FDR)                 : {n_bh}/{len(keys)}")
    print(f"  IC95 disjoints (rappel)                   : {n_disj}/{len(keys)}")
    # Paires qui basculent : disjointes mais NON significatives après Holm
    flips = [k for k, v in out_pairs.items()
             if v.get("ci95_disjoint") and not v["reject_holm"]]
    if flips:
        print("  ⚠ disjointes mais NON significatives après Holm :")
        for k in flips:
            v = out_pairs[k]
            print(f"      {k}  p_holm={v['p_holm']:.4f}")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
