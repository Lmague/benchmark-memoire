#!/usr/bin/env python3
"""Synthèse multi-modèles de la sonde tronquée par PCA.

Lit tous les `results/rapport_data/dim_probe_*.json` et produit
`dim_probe_summary.csv` + `dim_probe_summary.md` : F1 test par nombre de composantes
principales, pourcentage du F1 plein, et plus petit `k` atteignant 98 % / 99 % du plein.

Le résultat du manuscrit (« la tâche vit dans ~100 dimensions ») n'était établi que
pour DINOv3-B LVD ; cette synthèse le teste sur plusieurs architectures et familles.

    python3 scripts/rapport/dim_probe_summary.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(os.path.dirname(_HERE))

from registry import OUT  # noqa: E402


def main() -> None:
    files = sorted(glob.glob(os.path.join(OUT, "dim_probe_*.json")))
    files = [f for f in files if not f.endswith("summary.json")]
    rows = []
    for fp in files:
        model = os.path.basename(fp)[len("dim_probe_"):-len(".json")]
        pts = json.load(open(fp))
        if not pts:
            continue
        pts = sorted({int(p["k"]): p for p in pts}.values(), key=lambda p: p["k"])
        full = pts[-1]
        dim = full["k"]
        by_k = {int(p["k"]): float(p["test_f1"]) for p in pts}

        def k_for(frac):
            th = frac * float(full["test_f1"])
            for p in pts:
                if float(p["test_f1"]) >= th:
                    return int(p["k"])
            return dim

        rows.append({
            "model": model, "dim": dim, "f1_full": float(full["test_f1"]),
            **{f"f1_k{k}": by_k.get(k, "") for k in (10, 20, 50, 100, 200, 400)},
            "pct_k100": (100 * by_k.get(100, 0) / float(full["test_f1"])
                         if 100 in by_k else ""),
            "k_98pct": k_for(0.98), "k_99pct": k_for(0.99),
        })

    cols = ["model", "dim", "f1_full", "f1_k10", "f1_k20", "f1_k50", "f1_k100",
            "f1_k200", "f1_k400", "pct_k100", "k_98pct", "k_99pct"]
    out = os.path.join(OUT, "dim_probe_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    md = ["# Sonde tronquée par PCA — synthèse multi-modèles", "",
          "F1-macro test (11 classes) par nombre de composantes principales ; "
          "`k_98`/`k_99` = plus petit `k` atteignant 98 %/99 % du F1 plein. "
          "Sonde canonique (lbfgs, grille C réduite) ; mono-thread.", "",
          "| Modèle | dim | k=10 | k=20 | k=50 | k=100 | k=200 | k=400 | plein | % à k=100 | k_98 | k_99 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["f1_full"]):
        def g(k):
            v = r.get(f"f1_k{k}", "")
            return f"{v:.3f}" if v != "" else "—"
        md.append(f"| {r['model']} | {r['dim']} | {g(10)} | {g(20)} | {g(50)} | "
                  f"{g(100)} | {g(200)} | {g(400)} | {r['f1_full']:.3f} | "
                  f"{r['pct_k100']:.1f} | {r['k_98pct']} | {r['k_99pct']} |")
    md += ["", "Généré par `scripts/rapport/dim_probe_summary.py`."]
    with open(os.path.join(OUT, "dim_probe_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
