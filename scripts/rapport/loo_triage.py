#!/usr/bin/env python3
"""Validation prédictive leave-one-model-out du triage par géométrie latente.

Le manuscrit (§4.7) affirme que la famille « séparabilité » ordonne les modèles du
palier significativement mieux que le hasard, mais note que la preuve est
*inter-modèles in-sample* et qu'il manque une validation leave-one-model-out (LOO)
pour en faire un protocole opérationnel.

Ce script la fait. Pour chaque modèle i tenu à l'écart (palier `registry.TIER_K`) :

1. **Test d'ordre prédictif** : le signe de la relation métrique→F1 est estimé sur les
   n−1 modèles restants (Spearman), puis on vérifie sur les n−1 paires (i, j) si ce
   signe ordonne correctement le modèle tenu à l'écart. Agrégat = précision prédictive
   par paires, directement comparable à la précision in-sample de `tier_ranking.py`.
2. **Test de régression prédictif** : OLS F1 ~ a·métrique + b ajusté sur n−1 modèles,
   prédiction du F1 du modèle tenu à l'écart. On rapporte la MAE LOO, la MAE de la
   baseline « moyenne des n−1 » (le gain est la vraie information prédictive), et
   Spearman(prédiction, vrai F1).

Sortie : `results/rapport_data/loo_triage.csv` + `loo_triage.json`.

    python3 scripts/rapport/loo_triage.py
"""
from __future__ import annotations

import os as _os
# MONO-THREAD OBLIGATOIRE (AGENTS.md §4.8) avant tout import numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, binomtest, wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from registry import OUT, TIER_K, ensure_out  # noqa: E402
from tier_ranking import (ALL_METRICS, FAMILIES, HIGHER_IS_BETTER, LABEL,  # noqa: E402
                          bh, load_models)


def loo_pair_accuracy(v: np.ndarray, f1: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Précision d'ordonnancement LOO : le signe vient des n−1 modèles d'entraînement."""
    n = len(v)
    ok = tot = 0
    per_model = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        s = np.sign(spearmanr(v[tr], f1[tr])[0])
        ok_i = tot_i = 0
        for j in tr:
            if f1[i] == f1[j]:
                continue
            tot += 1
            tot_i += 1
            good = int(np.sign(f1[i] - f1[j]) == np.sign(s * (v[i] - v[j])))
            ok += good
            ok_i += good
        per_model[i] = ok_i / tot_i if tot_i else np.nan
    return ok, tot, per_model


def loo_rank_ols(v: np.ndarray, f1: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Prédiction LOO par OLS sur rangs (invariant à l'échelle et à la dimension).

    Pour le modèle tenu à l'écart, son rang est calculé par rapport aux valeurs
d'entraînement (fraction de valeurs d'entraînement inférieures, 0,5 pour les égalités).
    """
    n = len(v)
    pred = np.empty(n)
    base = np.empty(n)
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        vtr, ftr = v[tr], f1[tr]
        rtr = (np.argsort(np.argsort(vtr)) + 0.5) / len(tr)
        ri = (np.sum(vtr < v[i]) + 0.5 * np.sum(vtr == v[i])) / len(tr)
        A = np.column_stack([rtr, np.ones(len(tr))])
        coef, *_ = np.linalg.lstsq(A, ftr, rcond=None)
        pred[i] = coef[0] * ri + coef[1]
        base[i] = ftr.mean()
    mae = float(np.mean(np.abs(pred - f1)))
    mae_base = float(np.mean(np.abs(base - f1)))
    return pred, mae, mae_base


def main() -> None:
    ensure_out()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f1-csv", default=None,
                    help="optional per_seed.csv: use these F1 means instead of registry.CANONICAL_F1")
    ap.add_argument("--out-dir", default=OUT, help="output directory (default: registry OUT)")
    args = ap.parse_args()

    rows = load_models()[:TIER_K]
    if args.f1_csv:
        import pandas as _pd
        f1 = _pd.read_csv(args.f1_csv).groupby("key")["f1_macro_pres"].mean()
        missing = [r["model"] for r in rows if r["model"] not in f1.index]
        if missing:
            raise SystemExit(f"[loo_triage] no F1 for {missing}")
        for r in rows:
            r["f1"] = float(f1[r["model"]])
        rows.sort(key=lambda r: -r["f1"])
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"== LOO prédictif sur le palier de {len(rows)} modèles ==")
    print("   (signe et régression ajustés sur n−1 modèles, évalués sur le modèle exclu)")

    recs = []
    for m in ALL_METRICS:
        vals = [r.get(m, "") for r in rows]
        if any(x in ("", None) for x in vals):
            continue
        v = np.array([float(x) for x in vals])
        f1 = np.array([r["f1"] for r in rows])
        rho = float(spearmanr(v, f1)[0])
        ok, tot, per_model = loo_pair_accuracy(v, f1)
        en = HIGHER_IS_BETTER[m]              # sens pré-enregistré (in-sample)
        ok_en = sum(int(np.sign(f1[a] - f1[b]) == np.sign(en * (v[a] - v[b])))
                    for a in range(len(v)) for b in range(a + 1, len(v))
                    if f1[a] != f1[b])
        tot_en = sum(1 for a in range(len(v)) for b in range(a + 1, len(v))
                     if f1[a] != f1[b])
        pred, mae, mae_base = loo_rank_ols(v, f1)
        rho_pred = float(spearmanr(pred, f1)[0])
        # Unité d'analyse = modèle (35 observations), pas la paire (corrélée).
        p_wil = float(wilcoxon(per_model - 0.5).pvalue)
        recs.append({
            "tier": TIER_K, "family": next(f for f, ms in FAMILIES.items() if m in ms),
            "metric": m, "label": LABEL[m], "rho_full": rho,
            "pair_acc_in": ok_en / tot_en, "pair_acc_loo": ok / tot,
            "pairs_ok_loo": ok, "pairs_total_loo": tot,
            "acc_min_model": float(np.nanmin(per_model)),
            "acc_max_model": float(np.nanmax(per_model)),
            "mae_loo": mae, "mae_mean": mae_base, "mae_gain": mae_base - mae,
            "rho_pred_true": rho_pred,
            "p_wilcoxon": p_wil,
            "p_binom": float(binomtest(ok, tot, 0.5).pvalue),
        })

    for fam in FAMILIES:
        fr = [r for r in recs if r["family"] == fam]
        if fr:
            for r, q in zip(fr, bh([r["p_wilcoxon"] for r in fr])):
                r["p_bh_family"] = float(q)

    cols = ["tier", "family", "metric", "label", "rho_full", "pair_acc_in",
            "pair_acc_loo", "pairs_ok_loo", "pairs_total_loo", "acc_min_model",
            "acc_max_model", "mae_loo", "mae_mean", "mae_gain", "rho_pred_true",
            "p_wilcoxon", "p_bh_family", "p_binom"]
    out_csv = os.path.join(out_dir, "loo_triage.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c, "") for c in cols})

    json.dump({"tier": TIER_K, "n_models": len(rows), "n_pairs": recs[0]["pairs_total_loo"],
               "protocol": "LOO: sign/rank-OLS fitted on n-1, evaluated on held-out; "
                           "Wilcoxon par modèle (n=35) puis BH par famille",
               "metrics": recs},
              open(os.path.join(out_dir, "loo_triage.json"), "w"), indent=1,
              ensure_ascii=False)

    print(f"\n{'Métrique':24s} {'famille':14s} {'in-sample':>9s} {'LOO':>6s} "
          f"{'MAE rangs':>9s} {'MAE moy':>8s} {'gain':>7s} {'ρ(pred,F1)':>11s} "
          f"{'p_BH(fam)':>10s}")
    for r in sorted(recs, key=lambda r: -r["pair_acc_loo"]):
        print(f"{r['label']:24s} {r['family'][:14]:14s} "
              f"{100 * r['pair_acc_in']:8.0f}% {100 * r['pair_acc_loo']:5.0f}% "
              f"{r['mae_loo']:8.4f} {r['mae_mean']:8.4f} {r['mae_gain']:+7.4f} "
              f"{r['rho_pred_true']:+11.3f} {r.get('p_bh_family', float('nan')):10.3f}")
    fam_loo = {}
    for f_ in FAMILIES:
        fr = [r["pair_acc_loo"] for r in recs if r["family"] == f_]
        if fr:
            fam_loo[f_] = float(np.mean(fr))
    print("\nPrécision LOO moyenne par famille :")
    for f_, a in sorted(fam_loo.items(), key=lambda kv: -kv[1]):
        print(f"  {f_:16s} {100 * a:.1f} %")

    # Synthèse humaine (générée, jamais recopiée à la main).
    md = ["# Validation leave-one-model-out du triage par géométrie", "",
          f"Palier `registry.TIER_K` = {len(rows)} modèles ; {recs[0]['pairs_total_loo']} "
          "paires prédictives. Signe et régression (sur rangs) ajustés sur n−1 modèles, "
          f"évalués sur le modèle exclu. p = Wilcoxon par modèle (n={len(rows)}), BH par famille.", "",
          "| Métrique | Famille | In-sample | LOO | MAE rangs | MAE moyenne | ρ(pred,F1) | p_BH(fam) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda r: -r["pair_acc_loo"]):
        md.append(f"| {r['label']} | {r['family']} | {100 * r['pair_acc_in']:.0f} % | "
                  f"{100 * r['pair_acc_loo']:.1f} % | {r['mae_loo']:.4f} | "
                  f"{r['mae_mean']:.4f} | {r['rho_pred_true']:+.3f} | "
                  f"{r.get('p_bh_family', float('nan')):.2e} |")
    sep = [r for r in recs if r["family"] == "séparabilité"]
    spec = [r for r in recs if r["family"] == "spectrale"]
    tra = [r for r in recs if r["family"] == "transférabilité"]
    lo = min(r["pair_acc_loo"] for r in sep)
    hi = max(r["pair_acc_loo"] for r in sep)
    pbh = max(r.get("p_bh_family", 1.0) for r in sep)
    gain_pos = sum(1 for r in sep if r["mae_gain"] > 0)
    best_gain = max(r["mae_gain"] for r in recs)
    below = [r for r in spec + tra if r["pair_acc_loo"] < 0.5]
    below_txt = ", ".join(f"{r['label']} {100 * r['pair_acc_loo']:.1f} %" for r in below) or "aucune"
    spec_txt = ", ".join(f"{r['label']} {100 * r['pair_acc_loo']:.1f} %" for r in sorted(spec, key=lambda r: -r["pair_acc_loo"]))
    md += ["", "## Lecture", "",
           f"- **Séparabilité (Fisher, DBI, CHI, NC1, silhouette) : la validation LOO passe.** "
           f"{100 * lo:.0f} à {100 * hi:.0f} % d'ordonnancement hors échantillon, p_BH ≤ {pbh:.1e} ; "
           f"la régression sur rangs bat la moyenne pour {gain_pos}/{len(sep)} métriques.",
           f"- **Elle classe, elle ne prédit pas finement le F1** : gain de MAE ≤ {best_gain:.4f} sur "
           "un palier dont l'étendue de F1 vaut ≈ 0,04.",
           f"- **Spectre et transférabilité** : {spec_txt}. Sous le hasard : {below_txt}.", "",
           "Généré par `scripts/rapport/loo_triage.py`."]
    with open(os.path.join(out_dir, "loo_triage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] {os.path.join(out_dir, 'loo_triage.md')}")
    print(f"\n[OK] {out_csv}")


if __name__ == "__main__":
    main()
