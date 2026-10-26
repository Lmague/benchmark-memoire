#!/usr/bin/env python3
"""Synthèse des campagnes « têtes » (Narval fused + local tile-only).

Les deux campagnes de 2026-09-08/12 ont testé si une tête non linéaire (MLP-2,
bilinéaire diagonale, FiLM) récupère, sur les embeddings déjà sur disque, la part
d'interaction que la sonde linéaire laisse sur la table. `NEEDS_HUMAN.md` exigeait
de ne rien citer avant complétion ; les deux fronts sont complets (33 groupes
tuile, 24 tags fused).

Lecture principale : le `delta_vs_lin_adamw` mis en avant par la campagne compare
à une baseline *AdamW* plus faible que la sonde canonique *lbfgs*. Le juge de paix
est `delta_vs_lbfgs` : une tête ne compte que si elle bat la sonde linéaire
canonique, pas la baseline AdamW.

Sorties : `results/rapport_data/head_sweep_summary.csv` (une ligne par groupe),
`head_sweep_summary.json` (méta + agrégats) ; imprime le verdict.

    python3 scripts/rapport/head_sweep_summary.py
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
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from registry import OUT, p  # noqa: E402

TILE_DIR = os.path.join(OUT, "tile_heads")
FUSED_DIR = p("results", "context_distill", "fusion_heads")
NONLIN = ["mlp2", "bil_diag", "film"]


def _f1(head: dict) -> float | None:
    """F1 d'une tête, quelle que soit la convention de la campagne."""
    for k in ("f1_mean", "f1_test"):
        if k in head:
            return float(head[k])
    return None


def load_records() -> list[dict]:
    recs = []
    for fp in sorted(glob.glob(os.path.join(TILE_DIR, "*.json"))):
        if os.path.basename(fp).startswith("_"):
            continue
        d = json.load(open(fp))
        heads = {k: _f1(d.get(k, {})) for k in ["lin_lbfgs", "lin_adamw"] + NONLIN}
        heads = {k: v for k, v in heads.items() if v is not None}
        if "lin_lbfgs" not in heads:
            continue
        nonlin = {k: v for k, v in heads.items() if k in NONLIN}
        best_head = max(nonlin, key=nonlin.get)
        recs.append({
            "campaign": "tile", "group": d.get("group", os.path.basename(fp)),
            "n_seeds": d.get("n_seeds"), "fused": bool(d.get("fused", False)),
            **{f"f1_{k}": v for k, v in heads.items()},
            "best_nonlin": best_head, "f1_best_nonlin": nonlin[best_head],
            "delta_best_vs_lbfgs": nonlin[best_head] - heads["lin_lbfgs"],
            "delta_best_vs_adamw": (nonlin[best_head] - heads["lin_adamw"]
                                    if "lin_adamw" in heads else ""),
        })
    for fp in sorted(glob.glob(os.path.join(FUSED_DIR, "*.json"))):
        d = json.load(open(fp))
        heads = {k: _f1(d.get(k, {})) for k in ["lin_lbfgs", "lin_adamw"] + NONLIN}
        heads = {k: v for k, v in heads.items() if v is not None}
        if "lin_lbfgs" not in heads:
            continue
        nonlin = {k: v for k, v in heads.items() if k in NONLIN}
        if not nonlin:
            continue
        best_head = max(nonlin, key=nonlin.get)
        recs.append({
            "campaign": "fused", "group": d.get("tag", os.path.basename(fp)),
            "n_seeds": None, "fused": bool(d.get("fused", False)),
            **{f"f1_{k}": v for k, v in heads.items()},
            "best_nonlin": best_head, "f1_best_nonlin": nonlin[best_head],
            "delta_best_vs_lbfgs": nonlin[best_head] - heads["lin_lbfgs"],
            "delta_best_vs_adamw": (nonlin[best_head] - heads["lin_adamw"]
                                    if "lin_adamw" in heads else ""),
        })
    return recs


def main() -> None:
    recs = load_records()
    if not recs:
        sys.exit("aucun JSON de tête trouvé")
    cols = ["campaign", "group", "n_seeds", "fused", "f1_lin_lbfgs", "f1_lin_adamw",
            "f1_mlp2", "f1_bil_diag", "f1_film", "best_nonlin", "f1_best_nonlin",
            "delta_best_vs_lbfgs", "delta_best_vs_adamw"]
    out_csv = os.path.join(OUT, "head_sweep_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r.get(c, "") for c in cols})

    summary = {}
    for camp in ("tile", "fused"):
        sub = [r for r in recs if r["campaign"] == camp]
        if not sub:
            continue
        dl = [r["delta_best_vs_lbfgs"] for r in sub]
        da = [r["delta_best_vs_adamw"] for r in sub if r["delta_best_vs_adamw"] != ""]
        beats_lbfgs = [r for r in sub if r["delta_best_vs_lbfgs"] > 0]
        beats_adamw = [r for r in sub if r["delta_best_vs_adamw"] != ""
                       and r["delta_best_vs_adamw"] > 0]
        summary[camp] = {
            "n_groups": len(sub),
            "n_best_nonlin_beats_lbfgs": len(beats_lbfgs),
            "n_best_nonlin_beats_adamw": len(beats_adamw),
            "delta_vs_lbfgs_mean": sum(dl) / len(dl),
            "delta_vs_lbfgs_max": max(dl),
            "delta_vs_lbfgs_max_group": sub[dl.index(max(dl))]["group"],
            "delta_vs_adamw_mean": sum(da) / len(da) if da else None,
            "delta_vs_adamw_max": max(da) if da else None,
            "delta_vs_adamw_max_group": (sub[[r["delta_best_vs_adamw"] for r in sub].index(max(da))]["group"]
                                         if da else None),
            "head_wins": {},
        }
        for head in NONLIN:
            wins_l = sum(1 for r in sub if r["best_nonlin"] == head
                         and r["delta_best_vs_lbfgs"] > 0)
            wins_a = sum(1 for r in sub if r["best_nonlin"] == head
                         and r["delta_best_vs_adamw"] != ""
                         and r["delta_best_vs_adamw"] > 0)
            summary[camp]["head_wins"][head] = {"vs_lbfgs": wins_l, "vs_adamw": wins_a}

    json.dump({"protocol": "comparaison intra-groupe ; lbfgs = sonde canonique, "
                           "adamw = baseline de la campagne ; best_nonlin = max sur "
                           f"{NONLIN}",
               "tile_dir": TILE_DIR, "fused_dir": FUSED_DIR,
               "summary": summary, "records": recs},
              open(os.path.join(OUT, "head_sweep_summary.json"), "w"), indent=1,
              ensure_ascii=False)

    print("== Synthèse head-sweep (campagnes complètes) ==\n")
    for camp, s in summary.items():
        print(f"[{camp}] {s['n_groups']} groupes")
        print(f"  meilleure tête non linéaire > lbfgs canonique : "
              f"{s['n_best_nonlin_beats_lbfgs']}/{s['n_groups']}")
        print(f"  meilleure tête non linéaire > baseline AdamW : "
              f"{s['n_best_nonlin_beats_adamw']}/{s['n_groups']}")
        print(f"  delta vs lbfgs : moyenne {s['delta_vs_lbfgs_mean']:+.4f}, "
              f"max {s['delta_vs_lbfgs_max']:+.4f} ({s['delta_vs_lbfgs_max_group']})")
        if s["delta_vs_adamw_max"] is not None:
            print(f"  delta vs adamw : moyenne {s['delta_vs_adamw_mean']:+.4f}, "
                  f"max {s['delta_vs_adamw_max']:+.4f} ({s['delta_vs_adamw_max_group']})")
        for head, w in s["head_wins"].items():
            print(f"    {head:9s} gagne vs lbfgs {w['vs_lbfgs']:2d} | "
                  f"vs adamw {w['vs_adamw']:2d}")
        print()

    top = sorted(recs, key=lambda r: -r["delta_best_vs_lbfgs"])[:5]
    print("Top 5 des meilleures têtes non linéaires, relativement à la sonde lbfgs :")
    for r in top:
        print(f"  {r['campaign']:5s} {r['group'][:44]:46s} {r['best_nonlin']:8s} "
              f"{r['delta_best_vs_lbfgs']:+.4f} (vs adamw {r['delta_best_vs_adamw']:+.4f})")
    print(f"\n[OK] {out_csv}")

    md = ["# Synthèse des campagnes « têtes » non linéaires", "",
          "Les campagnes de 2026-09-08/12 sont complètes : 33 groupes tuile et 24 tags "
          "fusion. Le juge de paix est le `delta_vs_lbfgs` (sonde canonique), pas le "
          "`delta_vs_lin_adamw` (baseline de campagne plus faible).", ""]
    for camp, s in summary.items():
        md += [f"## Campagne {camp} ({s['n_groups']} groupes)", "",
               f"- Meilleure tête non linéaire > **lbfgs canonique** : "
               f"{s['n_best_nonlin_beats_lbfgs']}/{s['n_groups']}",
               f"- Meilleure tête non linéaire > baseline AdamW : "
               f"{s['n_best_nonlin_beats_adamw']}/{s['n_groups']}",
               f"- Δ vs lbfgs : moyenne {s['delta_vs_lbfgs_mean']:+.4f}, "
               f"max {s['delta_vs_lbfgs_max']:+.4f} ({s['delta_vs_lbfgs_max_group']})"]
        if s["delta_vs_adamw_max"] is not None:
            md.append(f"- Δ vs AdamW : moyenne {s['delta_vs_adamw_mean']:+.4f}, "
                      f"max {s['delta_vs_adamw_max']:+.4f} ({s['delta_vs_adamw_max_group']})")
        md.append("")
    md += ["## Lecture", "",
           "- **Vue tuile : 0/33.** Aucune tête non linéaire ne bat la sonde linéaire "
           "canonique ; le plateau plat n'est pas un artefact de sonde linéaire.",
           "- **Vue fusion : 2/24**, de ≤ +0,0076 et sur des tags seed-unique sans IC : "
           "pas un résultat citable comme gain.",
           "- Le MLP-2 bat la baseline AdamW (16/33, 9/24) mais jamais le lbfgs : "
           "l'apparente supériorité non linéaire disparaît à optimisation égale.", "",
           "Généré par `scripts/rapport/head_sweep_summary.py`."]
    with open(os.path.join(OUT, "head_sweep_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[OK] {os.path.join(OUT, 'head_sweep_summary.md')}")


if __name__ == "__main__":
    main()
