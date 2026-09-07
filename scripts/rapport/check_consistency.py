#!/usr/bin/env python3
"""Phase 0.e — Contrôle de cohérence avant compilation des rapports.

Vérifie que les données régénérées sous `results/rapport_data/` restent d'accord avec
les sources canoniques déclarées dans AGENTS.md §3 / results/README.md, et qu'aucune
valeur dépréciée (AGENTS.md §4.1) n'a été réintroduite dans les .tex.

Écrit `results/rapport_data/consistency_report.md` et renvoie un code de sortie non nul
en cas d'échec.

    python3 scripts/rapport/check_consistency.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from registry import (CANONICAL_F1, DEPRECATED, FORBIDDEN_VALUES,
                      NO_BOOTSTRAP_EMBEDDINGS, OUT, TIER_GROUP_KEY, TIER_K,
                      ensure_out, p)

CHECKS: list[dict] = []


def check(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def read_csv(fn):
    fp = os.path.join(OUT, fn)
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return list(csv.DictReader(f))


def c_canonical_f1():
    """Le probe canonique du dépôt reste la référence des tableaux maîtres."""
    d = json.load(open(p("results", "all_models_canonical_merged.json")))["models"]
    idx = {m["model"]: m for m in d}
    bad = []
    for k, (f1, _sd) in CANONICAL_F1.items():
        if k not in idx:
            bad.append(f"{k}: absent de all_models_canonical_merged.json")
            continue
        got = idx[k]["f1_linear_probe"]
        if abs(got - f1) > 5e-4:
            bad.append(f"{k}: registry {f1:.4f} ≠ canonique {got:.4f}")
    check("F1 canoniques du registre == all_models_canonical_merged.json",
          not bad, "; ".join(bad))


def c_datacurve_vs_canonical():
    """Le F1 du probe interne des runs diffère du probe canonique : le documenter."""
    rows = read_csv("screening_agg.csv")
    if rows is None:
        return check("screening_agg.csv présent", False, "fichier manquant")
    deltas = []
    for r in rows:
        if float(r["fraction"]) != 1.0:
            continue
        k = r["model"]
        if k in CANONICAL_F1:
            d = float(r["f1_pres_mean"]) - CANONICAL_F1[k][0]
            deltas.append((k, d))
    big = [(k, d) for k, d in deltas if abs(d) > 0.02]
    check("écart probe interne / probe canonique < 0,02 à 100 %",
          not big, "; ".join(f"{k}: Δ={d:+.4f}" for k, d in big) or
          f"max |Δ| = {max(abs(d) for _, d in deltas):.4f}")
    return deltas


def c_geometry_conventions():
    """eff_rank_sigma2 doit reproduire la colonne « RankMe » des rapports."""
    rows = read_csv("geometry_models.csv")
    if rows is None:
        return check("geometry_models.csv présent", False, "fichier manquant")
    ref = {"dinov3_vitb16_lvd": 61.4, "dinov3_vitl16_lvd": 58.5,
           "simdinov2_vitb16": 86.3, "simdinov2_vitl16": 98.7,
           "dinov3_vitb16_lvd_lora_r8": 43.7,
           "scalemae_vitl16": 10.9, "satmae_vitl16": 15.6}
    idx = {r["model"]: r for r in rows}
    bad = []
    for k, v in ref.items():
        if k not in idx:
            bad.append(f"{k}: absent")
            continue
        got = float(idx[k]["eff_rank_sigma2"])
        if abs(got - v) / v > 0.03:
            bad.append(f"{k}: attendu {v:.1f}, obtenu {got:.1f}")
    check("eff_rank_sigma2 == colonne RankMe de geometrie.tex (±3 %)",
          not bad, "; ".join(bad))


def c_lora_identity():
    """Le dossier d'embeddings de LoRA r=8 s'appelle `..._explora_frac100` (piège de
    nommage, AGENT_MEMORY.md 2026-07-22). Le discriminant fiable est le rang effectif :
    LoRA r=8 vaut 43,7 — c'est ce qui garantit qu'on n'a pas chargé un autre modèle."""
    rows = read_csv("geometry_models.csv")
    if rows is None:
        return
    a = {r["model"]: r for r in rows}.get("dinov3_vitb16_lvd_lora_r8")
    if not a:
        return check("identité LoRA r=8", False, "modèle absent")
    ra = float(a["eff_rank_sigma2"])
    check("LoRA r=8 identifié (eff_rank σ² ≈ 43,7)", abs(ra - 43.7) / 43.7 < 0.10,
          f"obtenu {ra:.1f}")


def c_tier_covered():
    """Le bootstrap apparié doit couvrir TOUT le palier compétitif.

    Le défaut qui motive ce contrôle : la campagne à 8 groupes laissait dehors trois
    membres du palier — dont le modèle n°1 — pendant que la prose affirmait un « tie
    statistique » en citant sa table."""
    fp = p("results", "significance_matrix_tier.json")
    if not os.path.exists(fp):
        return check("bootstrap du palier présent", False,
                     "significance_matrix_tier.json absent — lancer "
                     "scripts/rapport/significance_tier.py")
    d = json.load(open(fp))
    # Clés du palier (TIER_K premiers par F1 canonique)…
    ranked = sorted(CANONICAL_F1.items(), key=lambda kv: -kv[1][0])
    tier_keys = {k for k, _v in ranked[:TIER_K]}
    # …moins les modèles du palier sans embeddings rapatriés (point-estimates
    # documentés, jamais testés formellement)…
    expected = tier_keys - set(NO_BOOTSTRAP_EMBEDDINGS)
    # …comparés aux groupes du bootstrap via TIER_GROUP_KEY (display → clé).
    inv = {v: k for k, v in TIER_GROUP_KEY.items()}
    covered = {inv[g["name"]] for g in d["groups"] if g["name"] in inv}
    missing = sorted(expected - covered)
    extra = sorted(covered - tier_keys)
    n = len(d["groups"])
    check(f"le bootstrap apparié couvre le palier ({TIER_K} modèles, "
          f"sauf {len(tier_keys & set(NO_BOOTSTRAP_EMBEDDINGS))} sans embeddings)",
          not missing and not extra,
          (f"{n} groupes, {len(d['pairs'])} paires" if not missing and not extra
           else f"manquants={missing} ; hors-palier={extra}"))


def c_tier_reproduces_8group():
    """Contrôle de non-régression : le bootstrap du palier partage 8 groupes avec la
    campagne publiée et utilise les mêmes indices de rééchantillonnage (seed 42).
    Leurs F1 observés doivent donc coïncider. Un écart signale un changement de
    protocole silencieux — typiquement le nombre de threads BLAS (AGENTS.md §4.8),
    qui déplace un seed de 0,0015 sans que le solveur cesse de converger."""
    fp = p("results", "significance_matrix_tier.json")
    if not os.path.exists(fp):
        return
    new = {g["name"]: g["stats"]["observed"]
           for g in json.load(open(fp))["groups"]}
    old = {g["name"]: g["stats"]["observed"] for g in json.load(
        open(p("results", "significance_matrix_8group_fresh.json")))["groups"]}
    common = sorted(set(new) & set(old))
    bad = [f"{k}: {new[k]:.4f} vs {old[k]:.4f}"
           for k in common if abs(new[k] - old[k]) > 5e-4]
    check(f"les {len(common)} groupes communs reproduisent la campagne publiée",
          not bad and len(common) == 8,
          "; ".join(bad) or f"écart max {max((abs(new[k] - old[k]) for k in common), default=0):.5f}")


def c_no_explora():
    """Les deux régimes ExPLoRA sont exclus des rapports (registry.EXCLUDED_MODELS) :
    aucune trace ne doit subsister dans les sources LaTeX."""
    hits = []
    for pat in (p("rapport_bouguessa", "*.tex"),
                p("rapport_bouguessa", "tables", "*.tex")):
        for fp in sorted(glob.glob(pat)):
            txt = open(fp, encoding="utf-8", errors="replace").read()
            n = len(re.findall("explora", txt, flags=re.IGNORECASE))
            if n:
                hits.append(f"{os.path.basename(fp)}: {n}")
    check("aucune occurrence d'ExPLoRA dans les sources LaTeX", not hits,
          "; ".join(hits))


def c_no_deprecated_models():
    """AGENT_MEMORY.md §RÉSULTATS DÉPRÉCIÉS : vitb16_fulft_arctic, vitb16_arctic,
    resnet50_arctic sont des résultats seed-unique « à ne jamais citer ». Ils ne
    doivent apparaître dans aucun tableau des rapports, sous aucune graphie
    (underscore ou espace)."""
    hits = []
    needles = [k.replace("_", " ") for k in DEPRECATED] + list(DEPRECATED)
    for pat_glob in (p("rapport_bouguessa", "*.tex"),
                     p("rapport_bouguessa", "tables", "*.tex")):
        for fp in sorted(glob.glob(pat_glob)):
            txt = open(fp, encoding="utf-8", errors="replace").read()
            for n in needles:
                if re.search(re.escape(n), txt, flags=re.IGNORECASE):
                    hits.append(f"{os.path.basename(fp)}: {n}")
    check("aucun modèle déprécié (seed unique) dans rapport_bouguessa/*.tex",
          not hits, "; ".join(hits))


def c_forbidden_values():
    """AGENTS.md §4.1 : valeurs dépréciées interdites hors archive."""
    hits = []
    for fp in sorted(glob.glob(p("rapport_bouguessa", "*.tex"))):
        txt = open(fp, encoding="utf-8", errors="replace").read()
        for v in FORBIDDEN_VALUES:
            for pat in (v, v.replace(".", ",")):
                if re.search(re.escape(pat) + r"(?!\d)", txt):
                    hits.append(f"{os.path.basename(fp)}: {pat}")
    check("aucune valeur dépréciée dans rapport_bouguessa/*.tex", not hits,
          "; ".join(hits))


def c_figures_exist():
    """Toutes les figures référencées par les .tex doivent exister."""
    missing = []
    for fp in sorted(glob.glob(p("rapport_bouguessa", "*.tex"))):
        base = os.path.dirname(fp)
        txt = open(fp, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", txt):
            rel = m.group(1)
            cand = [os.path.join(base, rel)]
            if not os.path.splitext(rel)[1]:
                cand = [os.path.join(base, rel + e) for e in (".png", ".pdf", ".jpg")]
            if not any(os.path.exists(c) for c in cand):
                missing.append(f"{os.path.basename(fp)}: {rel}")
    check("toutes les figures référencées existent", not missing,
          "; ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))


def c_tiles():
    """Les effectifs par classe restent ceux du split canonique."""
    d = json.load(open(p("results", "tiles_per_class_per_split.json")))
    tot = sum(v["train"] for v in d.values())
    check("total tuiles train (12 classes) == 49 433", tot == 49433, f"obtenu {tot}")
    # Le schéma 11 classes retire RHOL : 49 433 − 152 = 49 281, la valeur qui
    # apparaît dans les courbes de données.
    check("total tuiles train (11 classes, sans RHOL) == 49 281",
          tot - d["RHOL"]["train"] == 49281, f"obtenu {tot - d['RHOL']['train']}")
    check("RHOL absente de val et test",
          d["RHOL"]["val"] == 0 and d["RHOL"]["test"] == 0,
          f"val={d['RHOL']['val']} test={d['RHOL']['test']}")
    n_test = sum(v["test"] for v in d.values())
    check("total tuiles test == 17 598", n_test == 17598, f"obtenu {n_test}")


def main():
    ensure_out()
    print("== Contrôles de cohérence ==")
    c_canonical_f1()
    deltas = c_datacurve_vs_canonical()
    c_geometry_conventions()
    c_lora_identity()
    c_tiles()
    c_forbidden_values()
    c_tier_covered()
    c_tier_reproduces_8group()
    c_no_explora()
    c_no_deprecated_models()
    c_figures_exist()

    n_ok = sum(c["ok"] for c in CHECKS)
    lines = ["# Contrôle de cohérence — `results/rapport_data/`", "",
             f"{n_ok}/{len(CHECKS)} contrôles passent.", "",
             "| Contrôle | Statut | Détail |", "|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c['name']} | {'✅' if c['ok'] else '❌'} | {c['detail'] or '—'} |")
    if deltas:
        lines += ["", "## Écart probe interne (run) vs probe canonique, à 100 %", "",
                  "| Modèle | Δ (interne − canonique) |", "|---|---|"]
        for k, d in sorted(deltas, key=lambda kv: -abs(kv[1])):
            lines.append(f"| `{k}` | {d:+.4f} |")
        lines += ["", "Les tableaux maîtres citent le **probe canonique** ; "
                  "les courbes de données citent le **probe interne du run** "
                  "(protocole homogène le long de la courbe)."]
    open(os.path.join(OUT, "consistency_report.md"), "w").write("\n".join(lines) + "\n")
    print(f"\n[RÉSULTAT] {n_ok}/{len(CHECKS)} — {os.path.join(OUT, 'consistency_report.md')}")
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
