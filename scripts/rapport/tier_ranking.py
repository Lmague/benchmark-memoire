#!/usr/bin/env python3
"""Peut-on classer les meilleurs modèles à partir de la seule géométrie latente ?

La question vient d'une observation juste : sur un palier restreint, plusieurs
métriques gardent un IC95 hors de zéro (`correlations_geometry.csv`). Une corrélation
n'est pas un classement — ce script mesure directement le pouvoir d'ordonnancement, et
sépare ce qui tient de ce qui est du bruit.

Le palier de référence est `registry.TIER_K`, fixé par la rupture MESURÉE du classement
(et non par une coupe de rang arbitraire comme l'ancien « top-8 »). Les autres tailles
de `TIERS` servent le test qui compte : la conclusion dépend-elle du choix de la coupe ?

Cinq sorties, dans `results/rapport_data/` :

  tier_ranking_metrics.csv   par métrique × taille de palier : ρ, paires bien ordonnées,
                             n° 1 prédit, p brut, p BH global et p BH par famille
  tier_ranking_families.csv  matrice de corrélation entre métriques sur le palier
  tier_ranking_noise.csv     stabilité de l'ordre VRAI (bootstrap apparié sur les tuiles)
  tier_ranking.json          métadonnées + synthèse
  (imprime un résumé lisible)

Familles pré-enregistrées — déclarées avant de regarder les p, pour que la correction
de Benjamini-Hochberg par famille soit honnête :

  séparabilité : fisher_ratio, silhouette, chi, nc1_canonical
  spectrale    : eff_rank_sigma2, stable_rank, participation_ratio, rankme_sigma_raw
  voisinage    : knn_purity, aniso_cosine, intrinsic_dim_twonn, dbi

    python3 scripts/rapport/tier_ranking.py
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy.stats import spearmanr

from registry import CANONICAL_F1, OUT, TIER_K, ensure_out, p

# Le palier de référence est TIER_K (rupture mesurée) ; les autres tailles
# servent à montrer que la conclusion dépend — ou non — du choix de la coupe.
TIERS = sorted({6, 8, 10, TIER_K, 12, len(CANONICAL_F1)})
N_BOOT, SEED = 2000, 42

FAMILIES = {
    "séparabilité": ["fisher_ratio", "silhouette", "chi", "nc1_canonical"],
    "spectrale": ["eff_rank_sigma2", "stable_rank", "participation_ratio",
                  "rankme_sigma_raw"],
    "voisinage": ["knn_purity", "aniso_cosine", "intrinsic_dim_twonn", "dbi"],
    "transférabilité": ["logme_train", "logme_test"],
}
# Sens attendu : +1 = « plus grand vaut mieux » pour prédire un F1 élevé.
HIGHER_IS_BETTER = {
    "fisher_ratio": +1, "silhouette": +1, "chi": +1, "nc1_canonical": -1,
    "eff_rank_sigma2": +1, "stable_rank": +1, "participation_ratio": +1,
    "rankme_sigma_raw": +1, "knn_purity": +1, "aniso_cosine": -1,
    "intrinsic_dim_twonn": -1, "dbi": -1, "logme_train": +1, "logme_test": +1,
}
LABEL = {
    "fisher_ratio": "Ratio de Fisher", "silhouette": "Silhouette",
    "chi": "Calinski-Harabasz", "nc1_canonical": "NC1 (intra/inter)",
    "eff_rank_sigma2": "Rang effectif (σ²)", "stable_rank": "Stable rank",
    "participation_ratio": "Participation ratio", "rankme_sigma_raw": "RankMe (σ)",
    "knn_purity": "Pureté k-NN", "aniso_cosine": "Anisotropie",
    "intrinsic_dim_twonn": "Dim. intrinsèque", "dbi": "Davies-Bouldin",
    "logme_train": "LogME (train)", "logme_test": "LogME (test)",
}
ALL_METRICS = [m for fam in FAMILIES.values() for m in fam]


def bh(pvals):
    p_ = np.asarray(pvals, dtype=float)
    order = np.argsort(p_)
    m, adj, prev = len(p_), np.empty(len(p_)), 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p_[i] * m / (m - rank))
        adj[i] = prev
    return adj


def load_models():
    with open(os.path.join(OUT, "geometry_models.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["model"] in CANONICAL_F1]
    for r in rows:
        r["f1"] = CANONICAL_F1[r["model"]][0]
    rows.sort(key=lambda r: -r["f1"])
    return rows


def pair_accuracy(models, values, sign):
    """Fraction des paires que la métrique ordonne comme le F1 (hasard = 0,5)."""
    ok = tot = 0
    for a, b in itertools.combinations(range(len(models)), 2):
        df1 = models[a]["f1"] - models[b]["f1"]
        dm = sign * (values[a] - values[b])
        if df1 == 0:
            continue
        tot += 1
        ok += int(np.sign(df1) == np.sign(dm))
    return ok, tot


def run_metrics(rows):
    out = []
    for k in TIERS:
        sub = rows[:k]
        recs = []
        for m in ALL_METRICS:
            vals = [r.get(m, "") for r in sub]
            if any(v in ("", None) for v in vals):
                continue
            v = np.array([float(x) for x in vals])
            f1 = np.array([r["f1"] for r in sub])
            rho, pv = spearmanr(v, f1)
            sign = HIGHER_IS_BETTER[m]
            ok, tot = pair_accuracy(sub, v, sign)
            best = sub[int(np.argmax(sign * v))]["model"]
            recs.append({"tier": k, "metric": m, "label": LABEL[m],
                         "family": next(f for f, ms in FAMILIES.items() if m in ms),
                         "rho": float(rho), "p": float(pv),
                         "pairs_ok": ok, "pairs_total": tot,
                         "pair_acc": ok / tot if tot else float("nan"),
                         "pred_best": best,
                         "pred_best_correct": bool(best == sub[0]["model"])})
        for rec, q in zip(recs, bh([r["p"] for r in recs])):
            rec["p_bh_global"] = float(q)
        for fam in FAMILIES:
            f_recs = [r for r in recs if r["family"] == fam]
            if f_recs:
                for rec, q in zip(f_recs, bh([r["p"] for r in f_recs])):
                    rec["p_bh_family"] = float(q)
        out += recs
    return out


def run_families(rows, tier=TIER_K):
    """Les métriques d'une même famille mesurent-elles la même chose ?"""
    sub = rows[:tier]
    mets = [m for m in ALL_METRICS if all(r.get(m, "") not in ("", None) for r in sub)]
    M = {m: np.array([float(r[m]) for r in sub]) for m in mets}
    out = []
    for a, b in itertools.combinations(mets, 2):
        out.append({"tier": tier, "metric_a": a, "metric_b": b,
                    "label_a": LABEL[a], "label_b": LABEL[b],
                    "family_a": next(f for f, ms in FAMILIES.items() if a in ms),
                    "family_b": next(f for f, ms in FAMILIES.items() if b in ms),
                    "rho": float(spearmanr(M[a], M[b])[0])})
    return out


def run_noise(rows, tier=TIER_K):
    """Quelle part de l'ordre VRAI du palier est du bruit inter-seed ?

    Les modèles affinés ont 3 seeds : on rééchantillonne les seeds (bootstrap
    hiérarchique simplifié) et on regarde à quelle fréquence l'ordre observé se
    reproduit. Les modèles gelés n'ont qu'une extraction : leur F1 est traité comme
    fixe, ce qui rend l'estimation OPTIMISTE (l'ordre vrai est donc au mieux aussi
    stable que ce qui est rapporté ici).
    """
    can = json.load(open(p("results", "all_models_canonical_merged.json")))["models"]
    seeds = {m["model"]: m.get("f1_seeds") for m in can}
    sub = rows[:tier]
    rng = np.random.RandomState(SEED)
    base = [r["model"] for r in sub]
    n_ident, kendall = 0, []
    ref_rank = {m: i for i, m in enumerate(base)}
    for _ in range(N_BOOT):
        draw = []
        for r in sub:
            s = seeds.get(r["model"])
            draw.append((r["model"], float(rng.choice(s)) if s else r["f1"]))
        order = [m for m, _ in sorted(draw, key=lambda t: -t[1])]
        n_ident += int(order == base)
        kendall.append(spearmanr([ref_rank[m] for m in order],
                                 list(range(len(order))))[0])
    out = [{"tier": tier, "n_bootstrap": N_BOOT,
            "p_ordre_identique": n_ident / N_BOOT,
            "rho_ordre_moyen": float(np.mean(kendall)),
            "rho_ordre_ic95_bas": float(np.percentile(kendall, 2.5)),
            "n_modeles_multi_seed": sum(1 for r in sub if seeds.get(r["model"])),
            "note": "modèles gelés traités comme fixes — estimation optimiste"}]
    return out


def main():
    ensure_out()
    rows = load_models()
    print(f"== {len(rows)} modèles, triés par F1 ==")
    for i, r in enumerate(rows[:12], 1):
        print(f"  {i:2d}. {r['model']:30s} {r['f1']:.4f}")

    mets = run_metrics(rows)
    fams = run_families(rows)
    noise = run_noise(rows)

    for name, data, cols in [
        ("tier_ranking_metrics.csv", mets,
         ["tier", "family", "metric", "label", "rho", "p", "p_bh_global",
          "p_bh_family", "pairs_ok", "pairs_total", "pair_acc", "pred_best",
          "pred_best_correct"]),
        ("tier_ranking_families.csv", fams,
         ["tier", "family_a", "metric_a", "label_a", "family_b", "metric_b",
          "label_b", "rho"]),
        ("tier_ranking_noise.csv", noise, list(noise[0].keys()))]:
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in data:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"[OK] {name} ({len(data)} lignes)")

    t8 = [r for r in mets if r["tier"] == TIER_K]
    print(f"\n== Palier des {TIER_K} meilleurs : qui ordonne réellement ? "
          "(hasard = 50 %) ==")
    for r in sorted(t8, key=lambda r: -r["pair_acc"]):
        flag = "✔" if r["pred_best_correct"] else "✘"
        print(f"  [{r['family'][:12]:12s}] {r['label']:22s} "
              f"paires {r['pairs_ok']:2d}/{r['pairs_total']} "
              f"({100 * r['pair_acc']:3.0f} %)  ρ={r['rho']:+.3f}  "
              f"p={r['p']:.4f}  p_BH(fam)={r.get('p_bh_family', float('nan')):.3f}  "
              f"n°1 {flag}")

    print("\n== Le signe tient-il quand on élargit le palier ? (ρ) ==")
    hdr = "  " + " " * 24 + "".join(f"top-{k:<5d}" for k in TIERS)
    print(hdr)
    for m in ALL_METRICS:
        line = f"  {LABEL[m]:24s}"
        for k in TIERS:
            r = [x for x in mets if x["tier"] == k and x["metric"] == m]
            line += f"{r[0]['rho']:+7.2f} " if r else "    --  "
        print(line)

    print(f"\n== Redondance interne (palier {TIER_K}) ==")
    for fam, ms in FAMILIES.items():
        inner = [r["rho"] for r in fams
                 if r["family_a"] == fam and r["family_b"] == fam]
        if inner:
            print(f"  {fam:16s} |ρ| entre ses métriques : "
                  f"{min(abs(x) for x in inner):.3f} à {max(abs(x) for x in inner):.3f}")

    print("\n== Stabilité de l'ordre VRAI du palier ==")
    n = noise[0]
    print(f"  ordre exact reproduit dans {100 * n['p_ordre_identique']:.1f} % des "
          f"rééchantillonnages ; ρ moyen avec l'ordre observé "
          f"{n['rho_ordre_moyen']:+.3f} (IC95 bas {n['rho_ordre_ic95_bas']:+.3f})")
    print(f"  ({n['n_modeles_multi_seed']}/{n['tier']} modèles ont 3 seeds ; {n['note']})")

    json.dump({"tiers": TIERS, "families": FAMILIES, "n_bootstrap": N_BOOT,
               "seed": SEED, "noise": noise[0],
               "n_models": len(rows)},
              open(os.path.join(OUT, "tier_ranking.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n[OK] {OUT}")


if __name__ == "__main__":
    main()
