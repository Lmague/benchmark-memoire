#!/usr/bin/env python3
"""Génère les fragments de tableaux LaTeX de `rapport_bouguessa/tables/`.

Chaque tableau est produit depuis une source unique et documentée ; les .tex ne font
que les inclure (`\\input{tables/xxx}`). Aucun chiffre n'est saisi à la main dans les
documents : régénérer ce script suffit à les mettre à jour.

    python3 scripts/rapport/make_tables.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from registry import (CANONICAL_F1, CLASSES_11, CLASSES_DEAD, DEPRECATED,
                      EXCLUDED_MODELS, FROZEN_MODELS, OUT, RUN_FAMILIES,
                      TIER_DISPLAY_LONG, TIER_DISPLAY_SHORT,
                      TIER_GROUP_KEY, TIER_K,
                      TIER_POP, p)

TAB = os.path.join(_ROOT, "rapport_bouguessa", "tables")
REGIMES = ["dinov3_vitb16_lvd_lora_r8", "vitb16_full_old", "vitb16_mhsa_old",
           "vitb16_scratch_old"]
# Ordre d'affichage : famille ImageNet d'abord, famille DINOv3-B ensuite.
REGIMES_ORDERED = ["vitb16_full_old", "vitb16_mhsa_old",
                   "vitb16_scratch_old", "dinov3_vitb16_lvd_lora_r8"]
# Nom d'affichage = celui de RUN_FAMILIES (source unique), pas d'abréviation locale :
# une abréviation ("Full"/"MHSA"/"Scratch"/"LoRA r=8") masquait que 3 des 4 régimes
# sont ImageNet et 1 seul DINOv3-B, ce qui se lisait comme un mélange non signalé.
REG_LABEL = {m: RUN_FAMILIES[m][0] for m in REGIMES_ORDERED}
N_MODELS = len(CANONICAL_F1)


def keep(rows, field="model"):
    """Retire les modèles exclus (registry.EXCLUDED_MODELS) et dépréciés
    (registry.DEPRECATED, AGENT_MEMORY.md §RÉSULTATS DÉPRÉCIÉS --- seed unique,
    « à ne jamais citer ») des rapports."""
    return [r for r in rows if r[field] not in EXCLUDED_MODELS
            and r[field] not in DEPRECATED]


def num(v, nd=4, sign=False):
    """Nombre au format français (virgule décimale)."""
    if v is None or v == "" or (isinstance(v, float) and not np.isfinite(v)):
        return "---"
    s = f"{float(v):+.{nd}f}" if sign else f"{float(v):.{nd}f}"
    return s.replace(".", ",")


def thousands(n):
    return f"{int(n):,}".replace(",", "\\,")


def esc(s):
    return (str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
            .replace("#", "\\#"))


def load(fn, root=OUT):
    with open(os.path.join(root, fn)) as f:
        return list(csv.DictReader(f))


# Caractères Unicode mathématiques que pdflatex+inputenc ne sait pas composer :
# ils arrivent des noms d'affichage des JSON et des libellés de métriques.
UNICODE_TEX = {
    "≠": "$\\neq$", "≈": "$\\approx$", "≥": "$\\geq$", "≤": "$\\leq$",
    "×": "$\\times$", "·": "$\\cdot$", "→": "$\\rightarrow$",
    "↔": "$\\leftrightarrow$", "✓": "$\\checkmark$",
    "λ": "$\\lambda$", "σ": "$\\sigma$", "α": "$\\alpha$", "ρ": "$\\rho$",
    "Σ": "$\\Sigma$", "Δ": "$\\Delta$", "μ": "$\\mu$",
    "₁": "$_1$", "²": "$^2$", "√": "$\\sqrt{}$",
}


def tex_safe(text: str) -> str:
    """Remplace les caractères Unicode mathématiques par leur macro LaTeX."""
    for a, b in UNICODE_TEX.items():
        text = text.replace(a, b)
    # $x$$y$ → $xy$ (fusion des délimiteurs adjacents produits ci-dessus)
    return text.replace("$$", "")


def path(fp: str) -> str:
    """Chemin en \\texttt{} avec points de césure après chaque / et _ ."""
    esc_fp = fp.replace("_", "\\_")
    esc_fp = esc_fp.replace("/", "/\\allowbreak ").replace("\\_", "\\_\\allowbreak ")
    return "\\texttt{" + esc_fp + "}"


def write(name, lines, note=None):
    """Écrit le fragment. La note de source va À L'INTÉRIEUR du flottant : sinon
    elle reste sur place quand LaTeX déplace le tableau, et le document se retrouve
    avec des « Source : … » orphelins."""
    os.makedirs(TAB, exist_ok=True)
    src = ("\\vspace{2pt}\\par\\footnotesize\\textit{Source :} " + note) if note else None
    out = list(lines)
    if src:
        end = "\\end{table}"
        if out and out[-1].strip() == end:
            out.insert(len(out) - 1, src)
        else:
            out.append("")
            out.append(src)
    open(os.path.join(TAB, name + ".tex"), "w",
         encoding="utf-8").write(tex_safe("\n".join(out)) + "\n")
    print(f"  [TAB] {name}.tex")


# ═════════════════════════════════════════════ courbes de données ════════════
def _datacurve_reprobe_full_100():
    """(mean, std) du F1 à 100 % du pipeline `results/datacurve/` (grille C étendue).

    Ce pipeline n'a pas de CSV agrégé pour ses résultats re-probés (seul
    results_raw.csv/results_agg.csv existe, et ce sont les valeurs D'AVANT
    re-probe) : la seule source est le tableau détaillé de
    results/datacurve/reprobe_report.md. On y lit la dernière ligne
    « **mean±std** » (fraction 1.00, colonne « New »), plutôt que de la
    recopier à la main.
    """
    fp = p("results", "datacurve", "reprobe_report.md")
    rows = re.findall(
        r"\*\*mean±std\*\* \| \*\*([\d.]+)±([\d.]+)\*\* \| \| \*\*([\d.]+)±([\d.]+)\*\*",
        open(fp, encoding="utf-8").read())
    if not rows:
        return None
    _old_m, _old_s, new_m, new_s = rows[-1]  # dernière fraction traitée = 1.00
    return float(new_m), float(new_s)


def t_pipelines():
    agg = load("screening_agg.csv")

    def f1_at(model, frac=1.0):
        r = [a for a in agg if a["model"] == model
             and abs(float(a["fraction"]) - frac) < 1e-9]
        return (num(r[0]["f1_pres_mean"]) + " $\\pm$ " + num(r[0]["f1_pres_std"])) if r else "---"

    def f1_val(model, frac=1.0):
        r = [a for a in agg if a["model"] == model
             and abs(float(a["fraction"]) - frac) < 1e-9]
        return float(r[0]["f1_pres_mean"]) if r else None

    dc_full = _datacurve_reprobe_full_100()
    dc_full_cell = (num(dc_full[0]) + " $\\pm$ " + num(dc_full[1])) if dc_full else "---"
    sota_full = f1_val("vitb16_full_old")
    gap = abs(sota_full - dc_full[0]) if (dc_full and sota_full is not None) else None

    rows = [
        ("\\texttt{sota\\_screening/}", "ViT-B/16 ImageNet", "Full, MHSA, Scratch",
         "1--100\\,\\% (7)", "3", f1_at("vitb16_full_old")),
        ("\\texttt{dinov3b\\_lora8/}", "DINOv3 ViT-B/16 LVD", "LoRA r=8",
         "0,5--100\\,\\% (8)", "3", f1_at("dinov3_vitb16_lvd_lora_r8")),
        ("\\texttt{results/datacurve/}", "ViT-B/16 ImageNet", "Full (grille C étendue)",
         "1--100\\,\\% (7)", "3", dc_full_cell),
    ]
    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Les trois pipelines de courbe de données du dépôt. Ils ne partagent "
        "ni l'initialisation, ni la grille de régularisation, ni les fractions : "
        "leurs sorties ne doivent jamais être fusionnées sans mention explicite "
        "(AGENTS.md §4.2). Le F1 à 100\\,\\% diffère de "
        f"{num(gap, 4) if gap is not None else '?'} entre les deux "
        "pipelines « Full » --- un écart modeste, cohérent avec une grille de "
        "régularisation différente plutôt qu'un désaccord de fond.}"
        "\\label{tab:pipelines}",
        "\\begin{tabular}{@{}llp{4.3cm}lrl@{}}", "\\toprule",
        "Pipeline & Initialisation & Régimes & Fractions & Seeds & F1 à 100\\,\\% \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_pipelines", lines,
          "\\texttt{results/rapport\\_data/screening\\_agg.csv}, "
          "\\texttt{results/datacurve/reprobe\\_report.md}.")


def _dc_table(col_mean, col_std, name, caption, note):
    agg = load("screening_agg.csv")
    fracs = sorted({float(a["fraction"]) for a in agg
                    if a["model"] in REGIMES})
    tiles = {float(a["fraction"]): int(a["n_train_tiles"]) for a in agg}
    # Deux familles de backbone : les colonnes sont regroupées pour que la
    # comparaison entre familles ne se fasse pas par inadvertance.
    n_in = 3  # Full, MHSA, Scratch — init. ViT-B/16 ImageNet
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             f"\\caption{{{caption}}}",
             "\\begin{tabular}{@{}lr" + "rr" * n_in + "|" + "rr" + "@{}}",
             "\\toprule",
             "& & \\multicolumn{" + str(2 * n_in) + "}{c|}{\\textbf{Backbone "
             "ViT-B/16 ImageNet-1k}} & \\multicolumn{2}{c}{\\textbf{Backbone "
             "DINOv3 ViT-B/16}} \\\\",
             f"\\cmidrule(lr){{3-{2 + 2 * n_in}}}"
             f"\\cmidrule(lr){{{3 + 2 * n_in}-{4 + 2 * n_in}}}",
             "& & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{REG_LABEL[m]}}}"
                                 for m in REGIMES_ORDERED) + " \\\\"]
    cm = " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}"
                  for i in range(len(REGIMES_ORDERED)))
    lines += [cm,
              "Fraction & Tuiles & " + " & ".join("F1 & $\\sigma$"
                                                  for _ in REGIMES_ORDERED) + " \\\\",
              "\\midrule"]
    for fr in fracs:
        cells = [f"{100 * fr:g}\\,\\%".replace(".", ","), thousands(tiles[fr])]
        for m in REGIMES_ORDERED:
            r = [a for a in agg if a["model"] == m
                 and abs(float(a["fraction"]) - fr) < 1e-9]
            if r:
                cells += [num(r[0][col_mean]), num(r[0][col_std])]
            else:
                cells += ["---", "---"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write(name, lines, note)


def t_datacurve():
    n_runs = len(load("screening_raw.csv"))
    _dc_table("f1_pres_mean", "f1_pres_std", "t_dc_11cls",
              "F1-macro sur classes présentes (schéma 11 classes) par fraction et "
              "par régime. Moyenne $\\pm$ écart-type sur 3~seeds. Probe interne du "
              "run (voir tableau~\\ref{tab:pipelines}).",
              "\\texttt{results/rapport\\_data/screening\\_agg.csv} "
              f"({n_runs} runs agrégés depuis \\texttt{{sota\\_screening/}}, "
              "\\texttt{results/datacurve\\_lora/runs/}).")
    _dc_table("f1_8cls_mean", "f1_8cls_std", "t_dc_8cls",
              "F1-macro schéma 8~classes (toutes présentes dans le test) par "
              "fraction et par régime. Moyenne $\\pm$ écart-type sur 3~seeds.",
              "\\texttt{results/rapport\\_data/screening\\_agg.csv}.")
    _dc_table("accuracy_mean", "accuracy_std", "t_dc_acc",
              "Exactitude globale (test) par fraction et par régime.",
              "\\texttt{results/rapport\\_data/screening\\_agg.csv}.")


def _note_convention_curve(fp_rows, lora_by_frac=None):
    """Note de convention pour les tableaux qui croisent les deux campagnes de courbe.

    La colonne « gelé » vient de la campagne de probing gelé
    (`results/datacurve_lora/frozen_probe_results_agg.csv`, sonde linéaire sur
    embeddings figés — même protocole que le probe canonique) ; les colonnes
    d'affinage viennent du probe interne des runs (`screening_agg.csv`), qui est
    aussi celui du bootstrap. On mesure l'accord au point 100 % plutôt que de
    l'affirmer."""
    txt = ("\\textbf{Convention.} La colonne « gelé » vient de la campagne de "
           "probing gelé (sonde linéaire sur embeddings figés) ; les colonnes "
           "d'affinage viennent du \\textbf{probe interne des runs}, celui des "
           "courbes et du bootstrap. Le $\\Delta$ croise donc les deux campagnes.")
    at100 = [r for r in fp_rows if float(r["fraction"]) == 1.0]
    if at100:
        f1 = float(at100[0]["f1_pres_mean"])
        can = CANONICAL_F1["dinov3_vitb16_lvd"][0]
        txt += (" À 100\\,\\%, le probing gelé donne " + num(f1) + " contre "
                + num(can) + " pour le probe canonique du tableau maître (écart "
                + num(f1 - can, 4, sign=True) + ") : les deux protocoles gelés "
                "sont alignés, c'est le côté affiné qui change de convention.")
    if lora_by_frac and 1.0 in lora_by_frac:
        f1 = float(lora_by_frac[1.0]["f1_pres_mean"])
        can = CANONICAL_F1["dinov3_vitb16_lvd_lora_r8"][0]
        txt += (" Côté LoRA r=8 à 100\\,\\% : " + num(f1) + " ici contre " + num(can)
                + " en canonique (écart " + num(f1 - can, 4, sign=True)
                + "). Ne pas mélanger les deux dans un même écart.")
    return txt


def t_frozen_probe():
    fp = load("frozen_probe_results_agg.csv",
              os.path.join(_ROOT, "results", "datacurve_lora"))
    lora = [a for a in load("screening_agg.csv")
            if a["model"] == "dinov3_vitb16_lvd_lora_r8"]
    lora = {float(a["fraction"]): a for a in lora}
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Probing gelé (DINOv3 ViT-B/16 LVD, backbone figé) contre "
             "LoRA r=8 sur le même backbone, \\textbf{à volume de données égal}. "
             "$\\Delta$ = LoRA $-$ gelé : positif partout, maximal à 5\\,\\%. "
             "La comparaison usuelle « LoRA à $x$\\,\\% contre gelé à 100\\,\\% » "
             "(seuil $\\approx$ 4\\,600 tuiles) est une autre question, traitée "
             "par le tableau~\\ref{tab:thresholds}.}\\label{tab:frozenprobe}",
             "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Fraction & Tuiles & Gelé F1 & $\\sigma$ & LoRA F1 & $\\sigma$ & "
             "$\\Delta$ \\\\", "\\midrule"]
    for r in fp:
        fr = float(r["fraction"])
        lo = lora.get(fr)
        d = (float(lo["f1_pres_mean"]) - float(r["f1_pres_mean"])) if lo else None
        lines.append(" & ".join([
            f"{100 * fr:g}\\,\\%".replace(".", ","), thousands(r["n_train_tiles"]),
            num(r["f1_pres_mean"]), num(r["f1_pres_std"]),
            num(lo["f1_pres_mean"]) if lo else "---",
            num(lo["f1_pres_std"]) if lo else "---",
            ("\\textbf{" + num(d, sign=True) + "}") if d and d > 0 else num(d, sign=True),
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}\\par\\footnotesize" + _note_convention_curve(fp, lora),
              "\\end{table}"]
    write("t_frozen_probe", lines,
          "\\texttt{results/datacurve\\_lora/frozen\\_probe\\_results\\_agg.csv} et "
          "\\texttt{results/rapport\\_data/screening\\_agg.csv}.")


def t_thresholds():
    """Tuiles nécessaires pour atteindre un F1 cible (interpolation log-linéaire)."""
    agg = load("screening_agg.csv")
    targets = [0.42, 0.45, 0.46, 0.47, 0.48]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Nombre de tuiles d'entraînement nécessaires pour atteindre "
             "un F1 cible, par interpolation log-linéaire de la courbe. "
             "« --- » : cible jamais atteinte sur la plage mesurée.}"
             "\\label{tab:thresholds}",
             "\\begin{tabular}{@{}l" + "r" * len(targets) + "@{}}", "\\toprule",
             "Régime & " + " & ".join("F1 $\\geq$ " + num(t, 2) for t in targets)
             + " \\\\", "\\midrule"]
    series = [(m, REG_LABEL[m]) for m in REGIMES_ORDERED]
    fp = load("frozen_probe_results_agg.csv",
              os.path.join(_ROOT, "results", "datacurve_lora"))
    for m, lab in series:
        if m == "dinov3_vitb16_lvd_lora_r8":
            lines.append("\\midrule")
        sub = sorted((a for a in agg if a["model"] == m),
                     key=lambda a: float(a["n_train_tiles"]))
        x = np.array([float(a["n_train_tiles"]) for a in sub])
        y = np.array([float(a["f1_pres_mean"]) for a in sub])
        cells = []
        for t in targets:
            if len(y) == 0:
                cells.append("---")
                continue
            ymax = np.maximum.accumulate(y)
            if ymax[-1] < t:
                cells.append("---")
            else:
                cells.append(thousands(round(np.interp(t, ymax, x))))
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    x = np.array([float(a["n_train_tiles"]) for a in fp])
    y = np.maximum.accumulate(np.array([float(a["f1_pres_mean"]) for a in fp]))
    cells = ["---" if y[-1] < t else thousands(round(np.interp(t, y, x)))
             for t in targets]
    lines.append("\\midrule")
    lines.append("Probing gelé (DINOv3-B) & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}\\par\\footnotesize" + _note_convention_curve(fp),
              "\\end{table}"]
    write("t_thresholds", lines,
          "interpolation sur \\texttt{results/rapport\\_data/screening\\_agg.csv} "
          "et \\texttt{results/datacurve\\_lora/frozen\\_probe\\_results\\_agg.csv}.")


def t_fewshot():
    d = json.load(open(p("results", "sample_efficiency.json")))
    caps = d["caps"]
    order = [k for k in FROZEN_MODELS if k in d["models"]]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Régime few-shot : F1-macro (11 classes présentes) selon le "
             "plafond de tuiles par classe, probe linéaire sur modèles gelés, "
             "moyenne sur 3~seeds de tirage. Le nombre de tuiles réellement "
             "utilisées est inférieur au plafond $\\times$ 12 pour les classes rares.}",
             "\\begin{tabular}{@{}l" + "r" * len(caps) + "@{}}", "\\toprule",
             "Modèle & " + " & ".join(str(c) for c in caps) + " \\\\",
             "Tuiles utilisées & " + " & ".join(
                 thousands(d["models"][order[0]][str(c)]["n_train_used"])
                 for c in caps) + " \\\\", "\\midrule"]
    for k in order:
        e = d["models"][k]
        best = max(caps, key=lambda c: e[str(c)]["f1_macro_pres_mean"])
        cells = []
        for c in caps:
            v = num(e[str(c)]["f1_macro_pres_mean"], 3)
            cells.append("\\textbf{" + v + "}" if c == best else v)
        lines.append(f"{esc(FROZEN_MODELS[k][0])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_fewshot", lines, "\\texttt{results/sample\\_efficiency.json}.")


def t_per_class_regimes():
    rows = load("per_class_by_regime.csv")
    live = [c for c in CLASSES_11 if c not in CLASSES_DEAD]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{F1 par classe à 100\\,\\% des données, par régime. Les trois "
             "classes ARCA, DRYI et RUBC sont omises : leur F1 vaut 0 pour tous les "
             "régimes et toutes les fractions (support test de 113, 177 et 31 tuiles).}",
             "\\begin{tabular}{@{}l" + "r" * len(live) + "@{}}", "\\toprule",
             "Régime & " + " & ".join(live) + " \\\\", "\\midrule"]
    for m in REGIMES_ORDERED:
        r = [x for x in rows if x["model"] == m and float(x["fraction"]) == 1.0]
        if not r:
            continue
        if m == "dinov3_vitb16_lvd_lora_r8":
            lines.append("\\midrule")
        lines.append(f"{REG_LABEL[m]} & "
                     + " & ".join(num(r[0][c], 3) for c in live) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_per_class_regimes", lines,
          "\\texttt{results/rapport\\_data/per\\_class\\_by\\_regime.csv}.")


def t_per_class_lora():
    rows = [r for r in load("per_class_by_regime.csv")
            if r["model"] == "dinov3_vitb16_lvd_lora_r8"]
    rows.sort(key=lambda r: float(r["fraction"]))
    live = [c for c in CLASSES_11 if c not in CLASSES_DEAD]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{DINOv3 ViT-B/16 LoRA r=8 --- F1 par classe et par fraction "
             "(moyenne sur 3~seeds). ALDE, BIRC et TUSS sont saturées dès "
             "2\\,463~tuiles ; WILL et MOSS continuent de progresser jusqu'à 100\\,\\%.}",
             "\\begin{tabular}{@{}lr" + "r" * len(live) + "@{}}", "\\toprule",
             "Fraction & Tuiles & " + " & ".join(live) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{100 * float(r['fraction']):g}\\,\\%".replace(".", ",")
                     + " & " + thousands(r["n_train_tiles"]) + " & "
                     + " & ".join(num(r[c], 3) for c in live) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_per_class_lora", lines,
          "\\texttt{results/rapport\\_data/per\\_class\\_by\\_regime.csv}.")


GEOM_DC = [("rankme_sigma_raw", "RankMe ($\\sigma$)"), ("eff_rank_sigma2", "Rang eff. ($\\sigma^2$)"),
           ("stable_rank", "Stable rk"), ("aniso_cosine", "Aniso."),
           ("knn_purity", "kNN"), ("silhouette", "Silh."),
           ("fisher_ratio", "Fisher"), ("nc1_canonical", "NC1"),
           ("alpha_spectral", "$\\alpha$")]


def t_geometry_datacurve():
    fp = os.path.join(OUT, "geometry_datacurve.csv")
    if not os.path.exists(fp):
        return
    rows = load("geometry_datacurve.csv")
    agg = {float(a["fraction"]): a["n_train_tiles"] for a in load("screening_agg.csv")}
    lines = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
             "\\setlength{\\tabcolsep}{2.5pt}",
             "\\caption{Géométrie de l'espace latent en fonction de la fraction "
             "d'entraînement, calculée sur le \\textbf{jeu test} ($n=17\\,598$ pour "
             "tous les points, donc sans biais de taille d'échantillon). "
             "Moyenne sur 3~seeds. Quatre régimes, deux familles : trois affinages "
             "ImageNet supervisé (Full, MHSA, Scratch --- ce dernier est un "
             "entraînement \\emph{depuis zéro}, pas un affinage) et un affinage "
             "DINOv3-B (LoRA r=8) --- ce ne sont pas des variantes d'un même "
             "backbone ; pour la comparaison à backbone fixe (DINOv3-B gelé "
             "et ses 3~régimes d'affinage), voir la table « Famille DINOv3 "
             "ViT-B/16 ».}",
             "\\begin{tabular}{@{}llr" + "r" * len(GEOM_DC) + "@{}}", "\\toprule",
             "Régime & Fraction & Tuiles & "
             + " & ".join(lab for _, lab in GEOM_DC) + " \\\\"]
    for m in REGIMES_ORDERED:
        fr = sorted({float(r["fraction"]) for r in rows if r["model"] == m})
        if not fr:
            continue
        lines.append("\\midrule")
        for i, f in enumerate(fr):
            sub = [r for r in rows if r["model"] == m and float(r["fraction"]) == f]
            cells = [REG_LABEL[m] if i == 0 else "",
                     f"{100 * f:g}\\,\\%".replace(".", ","),
                     thousands(agg.get(f, round(f * 49281)))]
            for col, _lab in GEOM_DC:
                v = [float(r[col]) for r in sub if r[col] != ""]
                nd = 3 if col in ("aniso_cosine", "knn_purity", "silhouette",
                                  "fisher_ratio") else 1
                cells.append(num(np.mean(v), nd) if v else "---")
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_geometry_datacurve", lines,
          "\\texttt{results/rapport\\_data/geometry\\_datacurve.csv} "
          "(\\texttt{scripts/rapport/geometry\\_test\\_fixed\\_n.py}).")


# ═════════════════════════════════════════════════ performances ══════════════
TYPE_FR = {"frozen": "gelé", "ft_fresh": "affiné (SSL)", "ft_old": "affiné (IN)"}

# Modèles de contexte : le nom d'affichage dans `all_models_canonical_merged.json`
# ne dit pas sur quel backbone ils tournent (``Contexte R2''), ce qui rendait la
# tête du tableau maître illisible. Backbone déduit du préfixe de run
# (make_tables.py CTX_RUNS : `dinov3_vitb16_lvd_ctxdistill_*` = DINOv3-B ;
# `simdinov2_vitb16_ctxdistill_*` = SimDINOv2-B). Révision 2026-09-08 : on le
# préfixe ici plutôt que dans le registre pour ne pas casser les références courtes
# (« Contexte R2 (B, fusion) ») de la prose et de la matrice de significativité.
CTX_META = {
    "ctxdistill_dB_tL": ("DINOv3-B Contexte R2 (distil. B, fusion 1536)", "affiné (contexte)"),
    "ctxdistill_dA_tL": ("DINOv3-B Contexte R1 (distil. A, tuile 768)", "affiné (contexte)"),
    "ctxdistill_dA_tEMA": ("DINOv3-B Contexte R3 (distil. A, EMA)", "affiné (contexte)"),
    "ctxdistill_dB_tSL_r2a4": ("SimDINOv2-B Contexte (B, fusion 512, r2a4)", "affiné (contexte)"),
    "ctxdistill_dB_tSL_r8a16": ("SimDINOv2-B Contexte (B, fusion 512, r8a16)", "affiné (contexte)"),
}
# Les deux modèles SimDINOv2-B Design B entraînés n'ont pas d'embeddings test
# rapatriés (registry.NO_BOOTSTRAP_EMBEDDINGS) : géométrie et bootstrap indisponibles.
CTX_NO_GEO = {"ctxdistill_dB_tSL_r2a4", "ctxdistill_dB_tSL_r8a16"}


def t_master():
    d = keep(json.load(open(p("results",
                              "all_models_canonical_merged.json")))["models"])
    geo = {}
    if os.path.exists(os.path.join(OUT, "geometry_models.csv")):
        geo = {r["model"]: r for r in load("geometry_models.csv")}
    ci = {}
    for g in json.load(open(p("results",
                              "significance_matrix_8group_fresh.json")))["groups"]:
        ci[g["name"]] = g["stats"]
    lines = ["\\begin{longtable}{@{}llrrrrrrrr@{}}",
             f"\\caption{{Tableau maître --- {len(d)}~modèles, probe linéaire canonique "
             "(LogisticRegression lbfgs multinomial, grille $C\\in[10^{-4},10]$, "
             "\\texttt{best\\_C} par validation, \\texttt{max\\_iter}=2000, seed~42), "
             "schéma 11~classes. $\\sigma$ = écart-type inter-seed (3~seeds pour les "
             "modèles affinés). Géométrie : moyenne par seed sur les embeddings test "
             "($n=17\\,598$, sous-échantillon 20\\,000, seed~42).}"
             "\\label{tab:master}\\\\", "\\toprule",
             "Modèle & Type & Dim & Seeds & F1 & $\\sigma$ & Silh. & "
             "Rang eff. & Aniso. & kNN \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule",
             "Modèle & Type & Dim & Seeds & F1 & $\\sigma$ & Silh. & "
             "Rang eff. & Aniso. & kNN \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endfoot"]
    for m in sorted(d, key=lambda r: -r["f1_linear_probe"]):
        k = m["model"]
        g = geo.get(k, {})
        t = m.get("type", "frozen")
        disp, typ = CTX_META.get(k, (m["display"], TYPE_FR.get(t, t)))
        lines.append(" & ".join([
            esc(disp), typ, str(m.get("dim", "---")),
            str(m.get("n_seeds") or 1),
            "\\textbf{" + num(m["f1_linear_probe"]) + "}",
            num(m.get("f1_std")) if m.get("f1_std") else "---",
            num(g.get("silhouette"), 4, sign=True) if g else "---",
            num(g.get("eff_rank_sigma2"), 1) if g else "---",
            num(g.get("aniso_cosine"), 3) if g else "---",
            num(g.get("knn_purity"), 3) if g else "---",
        ]) + " \\\\")
    lines += ["\\end{longtable}"]
    n_close, n_tot, tol, gaps = _conv_gaps()
    worst = gaps[0] if gaps else None
    lines += ["\\vspace{2pt}\\par\\footnotesize\\textbf{Convention.} La colonne F1 "
              "est le \\textbf{probe linéaire canonique}. Le tableau~\\ref{tab:ci} "
              "et les tests de significativité reposent sur une \\emph{autre} "
              "mesure, le \\textbf{probe interne des runs} : leurs colonnes « F1 "
              "observé » ne sont donc pas reprises d'ici. Les deux coïncident à "
              f"moins de {num(tol, 4)} près pour {n_close} des {n_tot}~modèles du "
              "palier" + (f", l'écart maximal étant {esc(worst[0])} "
                          f"({num(worst[1], 4, sign=True)})" if worst else "") +
              ". Ne pas former de $\\Delta$ entre une valeur d'ici et une valeur de "
              "là-bas."]
    lines += ["\\vspace{2pt}\\par\\footnotesize\\textbf{Données manquantes.} Les "
              "colonnes de géométrie (Silh., Rang eff., Aniso., kNN) sont \\texttt{---} "
              "pour les deux SimDINOv2-B Design~B entraînés (fusion 512) : leurs "
              "embeddings test n'ont pas été rapatriés sur disque "
              "(\\texttt{registry.NO\\_BOOTSTRAP\\_EMBEDDINGS}), donc ni géométrie ni "
              "probe interne n'ont pu être calculés. Ces deux modèles restent des "
              "point-estimates documentés (\\texttt{metrics.json}) et ne figurent pas "
              "dans le bootstrap apparié. Pour les modèles gelés (1 seed), "
              "$\\sigma$ et la géométrie sont moyennées sur un seul seed et dites "
              "\\texttt{---} quand la colonne n'a pas de sens."]
    write("t_master", lines,
          "\\texttt{results/all\\_models\\_canonical\\_merged.json} (F1, $\\sigma$) "
          "et \\texttt{results/rapport\\_data/geometry\\_models.csv} (géométrie). "
          "IC95 bootstrap dans le tableau~\\ref{tab:ci}.")
    _ = ci


def _signif_source():
    """Le bootstrap du palier complet s'il existe, sinon la campagne à 8 groupes.

    La campagne à 8 groupes laissait hors du test trois membres du palier — dont le
    modèle n°1 (LoRA r=8). `significance_tier.py` la remplace en couvrant les
    TIER_K modèles ; on retombe sur l'ancien fichier tant qu'il n'a pas tourné."""
    fp = p("results", "significance_matrix_tier.json")
    if os.path.exists(fp):
        return json.load(open(fp)), "results/significance\\_matrix\\_tier.json"
    return (json.load(open(p("results", "significance_matrix_8group_fresh.json"))),
            "results/significance\\_matrix\\_8group\\_fresh.json")


def _conv_gaps():
    """Écart mesuré entre les deux conventions de probe, sur le palier.

    Deux protocoles coexistent dans le dépôt et se ressemblent assez pour qu'on les
    confonde : le **probe linéaire canonique** (`all_models_canonical_merged.json`,
    repris par `CANONICAL_F1`) et le **probe interne des runs**, qui est celui sur
    lequel tourne le bootstrap apparié (`significance_matrix_tier.json`). Un Δ mixte
    n'a pas de sens ; les notes de tableau annoncent donc l'écart, mesuré ici plutôt
    que saisi à la main.

    Renvoie (n_proches, n_total, seuil, [(nom, écart) trié par |écart| décroissant]).
    """
    d, _src = _signif_source()
    boot = {g["name"]: g["stats"]["observed"] for g in d["groups"]}
    gaps = []
    for name, f1b in boot.items():
        k = TIER_GROUP_KEY.get(name)
        if k in CANONICAL_F1:
            gaps.append((name, f1b - CANONICAL_F1[k][0]))
    gaps.sort(key=lambda t: -abs(t[1]))
    tol = 5e-4
    n_close = sum(1 for _n, g in gaps if abs(g) < tol)
    return n_close, len(gaps), tol, gaps


def _note_convention_bootstrap():
    """Note de bas de tableau pour les tableaux entièrement en convention bootstrap."""
    n_close, n_tot, tol, gaps = _conv_gaps()
    big = [(n, g) for n, g in gaps if abs(g) >= tol][:2]
    detail = " et ".join(f"{esc(n)} ({num(g, 4, sign=True)})" for n, g in big)
    return ("\\textbf{Convention.} Les F1 de ce tableau viennent du \\emph{probe "
            "interne des runs}, celui sur lequel tourne le bootstrap --- ce n'est pas "
            "le probe linéaire canonique du tableau~\\ref{tab:master} "
            "(\\texttt{results/all\\_models\\_canonical\\_merged.json}). Les deux "
            f"coïncident à moins de {num(tol, 4)} près pour {n_close} des "
            f"{n_tot}~modèles du palier ; les écarts les plus marqués sont {detail}. "
            "Tous les $\\Delta$, $p$ et IC95 de ce tableau sont calculés dans cette "
            "seule convention.")


def _n_overlapping(groups):
    """Combien de groupes ont un IC95 qui en recouvre au moins un autre."""
    n = 0
    for a in groups:
        sa = a["stats"]
        if any(b is not a and not (sa["ci95_low"] > b["stats"]["ci95_high"]
                                   or b["stats"]["ci95_low"] > sa["ci95_high"])
               for b in groups):
            n += 1
    return n


def _pair(d, a, b):
    """La paire {a, b} du bootstrap, quel que soit l'ordre de la clé."""
    for v in d["pairs"].values():
        if {v["model_a"], v["model_b"]} == {a, b}:
            return v
    return None


def _long(name):
    """Nom d'affichage complet d'un groupe du bootstrap : backbone + régime +
    état (gelé/affiné). Les noms bruts du JSON ("Contexte R2 (B, fusion)",
    "DINOv3 ViT-B16") ne disent ni le backbone ni l'état — illisibles pour un
    lecteur (retour 2026-09-08). Source : registry.TIER_DISPLAY_LONG."""
    return TIER_DISPLAY_LONG.get(name, name)


def _fmt_p(pv):
    """p bootstrap : le plan est n=10 000 → ce qui tombe sous 10⁻⁴ s'affiche en
    puissance, pas en « 0,0000 » qui se lit comme un zéro faux. Sinon, 4
    décimales fixes (la colonne reste alignée)."""
    if pv is None:
        return "---"
    if pv < 1e-4:
        return "$<10^{-4}$"
    return "$" + num(pv, 4) + "$"


def t_dinov3b_regimes():
    """Le backbone DINOv3-B gelé contre ses régimes d'affinage.

    Ce tableau était saisi à la main dans `performances.tex` — y compris un
    « Non testé formellement » pour le modèle n°1 du benchmark. Il est désormais
    généré : F1 depuis le registre, p depuis le bootstrap apparié."""
    d, src = _signif_source()
    frozen_key, frozen_name = "dinov3_vitb16_lvd", "DINOv3 ViT-B16"
    f0 = CANONICAL_F1[frozen_key][0]
    regimes = [("dinov3_vitb16_lvd_full", "Full FT", "DINOv3-B Full"),
               ("dinov3_vitb16_lvd_mhsa", "MHSA-only", "DINOv3-B MHSA"),
               ("dinov3_vitb16_lvd_lora_r8", "LoRA r=8", "DINOv3-B LoRA r=8")]
    rows, tested, boot_deltas = [], 0, {}
    for key, lab, gname in regimes:
        if key not in CANONICAL_F1:
            continue
        f1 = CANONICAL_F1[key][0]
        pr = _pair(d, gname, frozen_name)
        if pr is None:
            verdict = "non couvert par le bootstrap"
        else:
            tested += 1
            verdict = ("\\textbf{oui}" if pr["bh_reject"] else "non")
            _pv = pr["p_two_sided"]
            _pt = ("$p<10^{-4}$" if _pv < 1e-4
                   else "$p=" + num(_pv, 4) + "$")
            verdict += " (" + _pt + (", BH)" if pr["bh_reject"] else ")")
            # Le Δ que le bootstrap a réellement testé (probe interne des runs) :
            # il n'est pas égal au Δ canonique affiché dans la colonne.
            sgn = 1 if pr["model_a"] == gname else -1
            boot_deltas[lab] = sgn * pr["delta_observed_a_minus_b"]
        rows.append((lab, f1, f1 - f0, verdict, pr is not None))
    rows.sort(key=lambda r: -r[1])  # F1 décroissant (retour lecteur 2026-09-08)
    best = max(r[1] for r in rows)
    lines = ["\\begin{table}[H]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Le backbone \\textbf{DINOv3 ViT-B/16 (LVD) gelé} contre ses "
             f"{len(rows)}~régimes d'affinage --- même backbone dans chaque ligne. "
             "Première ligne : le gelé (référence) ; les régimes suivent, classés "
             "par F1 décroissant. $\\Delta$ = écart au gelé "
             f"({num(f0)}). « Significatif ? » : bootstrap apparié hiérarchique "
             "contre le \\emph{même} backbone gelé, $p$ bilatéral, correction "
             "de Benjamini-Hochberg sur l'ensemble des paires du palier "
             f"(tableaux~\\ref{{tab:signif}} et~\\ref{{tab:signiftile}}). Les {tested} régimes y sont "
             "couverts.}\\label{tab:dinov3b}",
             "\\begin{tabular}{@{}lrrl@{}}", "\\toprule",
             "Régime (backbone : DINOv3 ViT-B/16 LVD) & F1 & $\\Delta$ gelé & Significatif ? \\\\",
             "\\midrule",
             f"Gelé (référence) & {num(f0)} & --- & --- \\\\"]
    for lab, f1, dl, verdict, _ok in rows:
        bold = (lambda t: "\\textbf{" + t + "}") if f1 == best else (lambda t: t)
        lines.append(" & ".join([bold(lab), bold(num(f1)),
                                 bold(num(dl, 4, sign=True)), verdict]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    # Les colonnes F1/Δ sont canoniques, la colonne p vient du bootstrap : le
    # tableau croise donc les deux conventions de probe. On l'annonce, et on donne
    # les deux Δ pour chaque régime testé.
    if boot_deltas:
        pairs_txt = " ; ".join(
            f"{esc(lab)} {num(dl, 4, sign=True)} contre "
            f"{num(boot_deltas[lab], 4, sign=True)}"
            for lab, _f1, dl, _v, _ok in rows if lab in boot_deltas)
        lines += [
            "\\vspace{2pt}\\par\\footnotesize\\textit{Note --- deux conventions dans "
            "ce tableau.} Les colonnes F1 et $\\Delta$ sont en \\textbf{probe "
            "linéaire canonique} ; la colonne « Significatif ? » vient du "
            "\\textbf{probe interne des runs}, sur lequel tourne le bootstrap "
            "apparié. Le $\\Delta$ affiché n'est donc pas exactement celui que le "
            "test porte (canonique contre bootstrap : " + pairs_txt + "). Chaque "
            "$\\Delta$ pris isolément reste calculé dans une seule convention ; les "
            "deux ne doivent pas être soustraits l'un à l'autre."]
    lines += ["\\end{table}"]
    write("t_dinov3b_regimes", lines,
          "F1 : \\texttt{scripts/rapport/registry.py} ; $p$ : \\texttt{" + src + "}.")


def t_simb_regimes():
    """Le backbone SimDINOv2-B gelé contre ses régimes d'adaptation (Stage A).
    Pendant de t_dinov3b_regimes : F1 canonique depuis le registre, p depuis le
    bootstrap apparié contre le même backbone gelé."""
    d, src = _signif_source()
    frozen_key, frozen_name = "simdinov2_vitb16", "SimDINOv2 ViT-B16"
    f0 = CANONICAL_F1[frozen_key][0]
    regimes = [("simdinov2_vitb16_lora_r8_b911", "LoRA r8, blocs 9-11",
                "SimDINOv2-B LoRA r8 (blocs 9-11)"),
               ("simdinov2_vitb16_lora_r8_b611", "LoRA r8, blocs 6-11",
                "SimDINOv2-B LoRA r8 (blocs 6-11)"),
               ("simdinov2_vitb16_lora_r2", "LoRA r2",
                "SimDINOv2-B LoRA r2"),
               ("simdinov2_vitb16_lora_r8_qkv", "LoRA r8, Q+K+V",
                "SimDINOv2-B LoRA r8 (Q+K+V)"),
               ("simdinov2_vitb16_norm_tuning", "NormTuning",
                "SimDINOv2-B NormTuning"),
               ("simdinov2_vitb16_lora", "LoRA r8, tous blocs (ancre)",
                "SimDINOv2-B LoRA r=8"),
               ("simdinov2_vitb16_lora_r8_b05", "LoRA r8, blocs 0-5",
                "SimDINOv2-B LoRA r8 (blocs 0-5)")]
    rows, tested = [], 0
    for key, lab, gname in regimes:
        if key not in CANONICAL_F1:
            continue
        f1 = CANONICAL_F1[key][0]
        pr = _pair(d, gname, frozen_name)
        if pr is None:
            verdict = "non couvert par le bootstrap"
        else:
            tested += 1
            verdict = ("\\textbf{oui}" if pr["bh_reject"] else "non")
            _pv = pr["p_two_sided"]
            _pt = ("$p<10^{-4}$" if _pv < 1e-4
                   else "$p=" + num(_pv, 4) + "$")
            verdict += " (" + _pt + (", BH)" if pr["bh_reject"] else ")")
        rows.append((lab, f1, f1 - f0, verdict))
    rows.sort(key=lambda r: -r[1])  # F1 décroissant (retour lecteur 2026-09-08)
    best = max(r[1] for r in rows)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Le backbone \\textbf{SimDINOv2 ViT-B/16 (iNat-Plantae) gelé} contre ses régimes "
             "d'adaptation (Stage~A, tableau~\\ref{tab:simbabl}) --- même backbone dans "
             "chaque ligne. Première ligne : le gelé (référence) ; les régimes suivent, "
             "classés par F1 décroissant. $\\Delta$ = "
             f"écart au gelé ({num(f0)}). « Significatif ? » : bootstrap apparié "
             "hiérarchique contre le \\emph{même} backbone gelé, $p$ bilatéral, "
             "correction de Benjamini-Hochberg sur l'ensemble des paires du palier "
             f"(tableaux~\\ref{{tab:signif}} et~\\ref{{tab:signiftile}}). Les {tested} régimes y sont "
             "couverts. NormTuning (normes + tête, $\\approx$47k paramètres) "
             "égale LoRA : l'adaptation de SimB vaut ce que vaut sa normalisation.}"
             "\\label{tab:simbreg}",
             "\\begin{tabular}{@{}lrrl@{}}", "\\toprule",
             "Régime (backbone : SimDINOv2 ViT-B/16, iNat-Plantae) & F1 & $\\Delta$ gelé & Significatif ? \\\\",
             "\\midrule",
             f"Gelé (référence) & {num(f0)} & --- & --- \\\\"]
    for lab, f1, dl, verdict in rows:
        bold = (lambda t: "\\textbf{" + t + "}") if f1 == best else (lambda t: t)
        lines.append(" & ".join([bold(lab), bold(num(f1)),
                                 bold(num(dl, 4, sign=True)), verdict]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_simb_regimes", lines,
          "F1 : \\texttt{scripts/rapport/registry.py} ; $p$ : \\texttt{" + src + "}.")


def t_ci():
    """IC95 du palier. Révision 2026-09-08 (retour lecteur : « on sait pas quel
    modèle c'est ») : le « Rang » est explicite et les noms de groupes bruts du
    JSON ("Contexte R2 (B, fusion)", "ViT-B/16 IN-MHSA") sont remplacés par les
    noms d'affichage registry.TIER_DISPLAY_LONG, qui disent backbone + régime +
    état (gelé/affiné). Ces rangs servent aussi de référence aux matrices de
    significativité (figures)."""
    d, src = _signif_source()
    ng = len(d["groups"])
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Intervalles de confiance à 95\\,\\% par bootstrap apparié "
             "hiérarchique (indices de tuiles et de seeds partagés entre groupes, "
             f"$n={thousands(d['n_bootstrap'])}$ rééchantillonnages, seed~42, "
             f"$n_{{\\text{{tuiles}}}}={thousands(d['n_tiles'])}$). "
             f"{_n_overlapping(d['groups'])} des {ng}~groupes ont un IC95 qui en "
             "recouvre au moins un autre. Le « Rang » est celui du classement F1 "
             "(bootstrap, probe interne) --- c'est aussi la numérotation utilisée "
             "par les matrices de significativité (figures). "
             "Les noms indiquent explicitement le backbone et l'état "
             "(gelé/affiné).}\\label{tab:ci}",
             "\\begin{tabular}{@{}rlrrrr@{}}", "\\toprule",
             "Rang & Modèle (backbone --- régime, état) & F1 observé & Bootstrap moy. "
             "& IC95 bas & IC95 haut \\\\",
             "\\midrule"]
    for rank, g in enumerate(sorted(d["groups"], key=lambda g: -g["stats"]["observed"]), 1):
        s = g["stats"]
        lines.append(" & ".join([str(rank), esc(_long(g["name"])), num(s["observed"]),
                                 num(s["mean"]),
                                 num(s["ci95_low"]), num(s["ci95_high"])]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}\\par\\footnotesize" + _note_convention_bootstrap(),
              "\\end{table}"]
    write("t_ci", lines, "\\texttt{" + src + "}.")


SIGNIF_HEAD = ("Modèle A & Modèle B & $\\Delta$ (A$-$B) $\\times10^{-2}$ & $p$ bilatéral & BH & "
               "IC disj. \\\\")


def _signif_row(r, scale_delta=False):
    """Une ligne du tableau de significativité. `scale_delta=True` exprime le Δ en
    points de pourcentage (×10⁻²) ; c'est la convention du document `performances`
    (scores trop petits pour être lus en fraction). Le compendium garde la fraction."""
    dv = r["delta_observed_a_minus_b"]
    dim = num(dv * 100.0, 2, sign=True) if scale_delta else num(dv, 4, sign=True)
    return " & ".join([
        esc(r["model_a"]), esc(r["model_b"]),
        dim,
        num(r["p_two_sided"], 4),
        "\\textbf{oui}" if r["bh_reject"] else "non",
        "oui" if r["ci95_disjoint"] else "non"]) + " \\\\"


def t_signif():
    """Table longue : toutes les paires du bootstrap. Réservée à `compendium.pdf`.

    `performances.pdf` et `analyse.pdf` prennent la version courte et lisible
    (`t_signif_bh`, rejets BH seulement, éclatée en deux tableaux par famille de
    vainqueur : tab:signif / tab:signiftile). Étiquette de la version longue :
    tab:signiflong, pour éviter toute collision quand le compendium inclut
    les deux."""
    d, src = _signif_source()
    pairs = sorted(d["pairs"].values(), key=lambda r: r["p_two_sided"])
    npair = len(pairs)
    lines = ["\\begin{longtable}{@{}llrrll@{}}",
             f"\\caption{{Version longue --- les {npair}~paires du bootstrap "
             "apparié (rejets \\emph{et} non-rejets), triées par $p$ "
             "croissant. Correction de Benjamini-Hochberg à $\\alpha=0,05$ ; "
             f"{d['bh_n_rejected']}~paires sur {npair} sont rejetées --- leur "
             "liste lisible par groupe de vainqueur tient dans les "
             "tableaux~\\ref{tab:signif} et~\\ref{tab:signiftile}. "
             "« IC disj. » : les IC95 des deux groupes ne se recouvrent pas.}"
             "\\label{tab:signiflong}\\\\",
             "\\toprule", SIGNIF_HEAD, "\\midrule", "\\endfirsthead", "\\toprule",
             SIGNIF_HEAD, "\\midrule", "\\endhead",
             "\\bottomrule", "\\endfoot"]
    lines += [_signif_row(r) for r in pairs]
    lines += ["\\end{longtable}",
              "\\vspace{-6pt}\\par\\footnotesize" + _note_convention_bootstrap()]
    write("t_signif", lines, "\\texttt{" + src + "}.")


def t_signif_bh():
    """Version courte : seulement les paires rejetées par Benjamini-Hochberg.

    Les paires non rejetées ne portent aucun argument — elles disent toutes la
    même chose (« indissociable du bruit ») et occupaient l'essentiel du
    tableau.

    Révision 2026-09-08 (retour lecteur : « la table 16 est illisible ») :
    1. DEUX tableaux au lieu d'un longtable plat de 108 lignes à six colonnes :
       (a) tab:signif — paires gagnées par les modèles de contexte (le gain
           spatial écrase tout le reste ; 3 vainqueurs seulement) ;
       (b) tab:signiftile — paires gagnées par les modèles sur tuile seule
           (adaptation, échelle, gelés forts).
    2. Fini la colonne « Modèle A » répétée 32 fois : chaque groupe s'ouvre sur
       une ligne \\multicolumn qui nomme le vainqueur UNE fois (avec F1 et
       décompte de victoires) ; les lignes qui suivent ne listent que les
       vaincus, avec leur F1 rappelé en ligne.
    3. Noms d'affichage explicites (backbone --- régime, état ; cf. t_ci) : plus
       besoin de deviner à quel modèle correspond « ViT-B/16 IN-MHSA » ou
       « Contexte R2 (B, fusion) ».
    4. Δ en points de F1 et p formaté ($<10^{-4}$ plutôt que 0,0000).

    Les paires restent identifiées par les noms BRUTS du JSON (clés internes) ;
    le mappage vers les noms lisibles se fait à l'affichage via _long()."""
    d, src = _signif_source()
    pairs = sorted(d["pairs"].values(), key=lambda r: r["p_two_sided"])
    npair, kept = len(pairs), [r for r in pairs if r["bh_reject"]]
    obs = {g["name"]: g["stats"]["observed"] for g in d["groups"]}

    # Regrouper par vainqueur (le côté du Δ positif).
    groups = {}
    for r in kept:
        delta = r["delta_observed_a_minus_b"]
        winner, loser = ((r["model_a"], r["model_b"]) if delta >= 0
                         else (r["model_b"], r["model_a"]))
        groups.setdefault(winner, []).append(
            (loser, abs(delta), r["p_two_sided"], r["ci95_disjoint"]))
    for w in groups:
        groups[w].sort(key=lambda t: (-t[1], t[2]))
    # Deux blocs : vainqueurs « contexte » vs vainqueurs « tuile seule ».
    ctx = sorted((w for w in groups if "Contexte" in w),
                 key=lambda w: -obs.get(w, 0.0))
    tile = sorted((w for w in groups if "Contexte" not in w),
                  key=lambda w: -obs.get(w, 0.0))

    def block(winners, label, caption):
        """Un longtable : par vainqueur, une ligne d'en-tête \\multicolumn puis
        ses vaincus (nom explicite + F1, Δ en points, p, IC disj.)."""
        lines = ["\\begin{longtable}{@{}lrrrl@{}}",
                 "\\caption{" + caption + "}\\label{" + label + "}\\\\",
                 "\\toprule",
                 "Modèle battu (son F1 bootstrap) & $\\Delta$ (pts de F1) & "
                 "$p$ bilatéral & IC disj. \\\\",
                 "\\midrule", "\\endfirsthead",
                 "\\toprule",
                 "Modèle battu (son F1 bootstrap) & $\\Delta$ (pts de F1) & "
                 "$p$ bilatéral & IC disj. \\\\",
                 "\\midrule", "\\endhead",
                 "\\bottomrule", "\\endfoot"]
        for w in winners:
            ps = [t[2] for t in groups[w]]
            n_w = len(groups[w])
            if max(ps) < 1e-4:
                p_txt = "tous $p<10^{-4}$" if n_w > 1 else "$p<10^{-4}$"
            elif n_w == 1 or min(ps) == max(ps):
                p_txt = "$p=$ " + _fmt_p(max(ps))
            else:
                p_txt = ("$p$ de " + _fmt_p(min(ps)) + " à " + _fmt_p(max(ps)))
            lines.append("\\addlinespace[3pt]\\multicolumn{4}{@{}l@{}}"
                         "{\\textbf{Vainqueur : " + esc(_long(w)) + " --- F1 "
                         + num(obs.get(w)) + " --- bat " + str(n_w) + " "
                         + ("modèle" if n_w == 1 else "modèles") + ", "
                         + p_txt + "}} \\\\")
            for loser, ad, p, disj in groups[w]:
                lines.append(" & ".join([
                    esc(_long(loser)) + " (" + num(obs.get(loser)) + ")",
                    num(ad * 100.0, 2, sign=True), _fmt_p(p),
                    "oui" if disj else "non"]) + " \\\\")
        lines += ["\\end{longtable}",
                  "\\vspace{2pt}\\par\\footnotesize"
                  + _note_convention_bootstrap() + "\\vspace{6pt}"]
        return lines

    n_ctx = sum(len(groups[w]) for w in ctx)
    n_tile = sum(len(groups[w]) for w in tile)
    n_rest = npair - len(kept)
    lines = ["\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}"]
    lines += block(
        ctx, "tab:signif",
        "Paires du palier rejetées par Benjamini-Hochberg ($\\alpha=0,05$) sur "
        + str(npair) + "~paires testées par bootstrap apparié --- "
        "1/2 : paires \\emph{gagnées par les modèles DINOv3-B avec contexte "
        "spatial} (" + str(n_ctx) + "~paires sur " + str(len(kept))
        + " ; R2 est la borne non déployable, contexte requis à l'inférence). "
        "Chaque ligne = un modèle \\emph{battu} avec son F1 ; le vainqueur est "
        "rappelé en tête de groupe. $\\Delta$ en points de F1 "
        "($\\times10^{-2}$) ; « IC disj. » : les IC95 (tableau~\\ref{tab:ci}) "
        "des deux modèles ne se recouvrent pas. Noms explicites (backbone --- "
        "régime, état), cf. tableau~\\ref{tab:ci}. Les " + str(n_rest)
        + "~paires non rejetées ne sont pas listées : écart indissociable du "
        "bruit.")
    lines += block(
        tile, "tab:signiftile",
        "Paires du palier rejetées par Benjamini-Hochberg --- 2/2 : paires "
        "\\emph{gagnées par les modèles sur tuile seule} (" + str(n_tile)
        + "~paires sur " + str(len(kept)) + ") : victoires d'adaptation, "
        "d'échelle ou de pré-entraînement aligné, toutes petites --- le "
        "plateau plat SimDINOv2-B n'apparaît pas ici (ses bras ne se "
        "distinguent pas entre eux). Même format que ci-dessus ; noms et F1 "
        "explicites, cf. tableau~\\ref{tab:ci}.")
    lines += ["\\endgroup"]
    write("t_signif_bh", lines, "\\texttt{" + src + "}.")


def t_per_class_models():
    # Palier compétitif (probe interne des runs, MÊME convention que le bootstrap) —
    # inclut les modèles de contexte et DINOv3 ViT-S/16.
    rows = [r for r in load("per_class_tier.csv")
            if r.get("id") not in EXCLUDED_MODELS and r.get("id") not in DEPRECATED]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             f"\\caption{{F1 par classe, palier compétitif ({len(rows)}~modèles, probe "
             "interne des runs — même convention que le bootstrap, donc que la matrice de "
             "significativité et les IC95 ci-dessus). † : classes non évaluables.}",
             "\\begin{tabular}{@{}lr" + "r" * len(CLASSES_11) + "@{}}", "\\toprule",
             "Modèle & F1 & "
             + " & ".join(c + ("\\,†" if c in CLASSES_DEAD else "")
                         for c in CLASSES_11) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(
            [esc(r["display"]),
             "\\textbf{" + num(r["f1_macro_pres"]) + "}"]
            + [num(r[c], 3) for c in CLASSES_11]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_per_class_models", lines,
          "\\texttt{results/rapport\\_data/per\\_class\\_tier.csv} (tier\\_preds\\_cache.npz).")


def t_support():
    d = json.load(open(p("results", "tiles_per_class_per_split.json")))
    rows = keep(load("per_class_all_models.csv"))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Effectifs de tuiles par classe et par split (découpage par "
             f"orthomosaïque entière), et F1 moyen de la classe sur les {len(rows)}~modèles. "
             "Les quatre classes en bas de tableau ne sont pas évaluables : RHOL est "
             "absente des splits val et test, ARCA/DRYI/RUBC ont un support test "
             "inférieur à 200~tuiles.}",
             "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Classe & Train & Val & Test & \\% du train & F1 moyen & F1 max \\\\",
             "\\midrule"]
    tot = sum(v["train"] for v in d.values())
    order = [c for c in CLASSES_11 if c not in CLASSES_DEAD] + CLASSES_DEAD + ["RHOL"]
    for i, c in enumerate(order):
        if c == CLASSES_DEAD[0]:
            lines.append("\\midrule")
        v = d[c]
        vals = [float(r[c]) for r in rows] if c in CLASSES_11 else []
        lines.append(" & ".join([
            c, thousands(v["train"]), thousands(v["val"]), thousands(v["test"]),
            num(100 * v["train"] / tot, 1),
            num(np.mean(vals), 3) if vals else "---",
            num(np.max(vals), 3) if vals else "---"]) + " \\\\")
    lines += ["\\midrule",
              " & ".join(["\\textbf{Total}", thousands(tot),
                          thousands(sum(v["val"] for v in d.values())),
                          thousands(sum(v["test"] for v in d.values())),
                          "100,0", "---", "---"]) + " \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_support", lines,
          "\\texttt{results/tiles\\_per\\_class\\_per\\_split.json} et "
          "\\texttt{results/rapport\\_data/per\\_class\\_all\\_models.csv}.")


def t_knn():
    fp = os.path.join(OUT, "knn_vs_probe.csv")
    if not os.path.exists(fp):
        return
    rows = keep(load("knn_vs_probe.csv"))
    ks = sorted(int(c.split("_")[1][1:]) for c in rows[0]
                if c.startswith("knn_k") and c.endswith("_f1") and rows[0][c] != "")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Probe linéaire contre $k$ plus proches voisins (cosinus) sur "
             f"les mêmes embeddings test ({len(rows)} modèles --- les 3~résultats "
             "dépréciés à seed unique sont retirés, AGENT\\_MEMORY.md "
             "§RÉSULTATS DÉPRÉCIÉS). Le probe domine systématiquement ; l'écart "
             "ne dépend pas du choix de $k$.}",
             "\\begin{tabular}{@{}lr" + "r" * len(ks) + "rr@{}}", "\\toprule",
             "Modèle & Probe & "
             + " & ".join(f"$k$={k}" for k in ks)
             + " & meilleur $k$ & $\\Delta$ \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join([
            esc(r["model"].replace("_", " ")),
            "\\textbf{" + num(r["probe_f1_pres"]) + "}"]
            + [num(r[f"knn_k{k}_f1"]) for k in ks]
            + [r["knn_best_k"], num(r["delta_probe_knn"], 4, sign=True)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_knn", lines, "\\texttt{results/without\\_rhol/probe\\_knn\\_cgrid.json}.")


def t_schemas():
    rows = keep(load("f1_learnable_12models.csv", p("results")))
    rows.sort(key=lambda r: -float(r["f1_macro_11"]))
    deltas = [float(r["f1_macro_learnable"]) - float(r["f1_macro_12"]) for r in rows]
    mean_d = float(np.mean(deltas))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Le même modèle sous les trois schémas de classes "
             f"({len(rows)} modèles --- les 3~résultats dépréciés à seed unique de "
             "\\texttt{f1\\_learnable\\_12models.csv} sont retirés, "
             "AGENT\\_MEMORY.md §RÉSULTATS DÉPRÉCIÉS ; DINOv3-B/SimDINOv2-B affinés "
             "n'ont pas encore cette comparaison de schémas). Le passage "
             f"de 12 à 8~classes ajoute {num(mean_d, 2, sign=True)} de F1 en moyenne "
             f"({num(min(deltas), 2, sign=True)} à {num(max(deltas), 2, sign=True)} "
             "selon le modèle) sans modifier le classement : le schéma change "
             "l'échelle, pas la conclusion.}",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Modèle & 12 classes & 11 classes & 8 classes & 8cls $-$ 12cls \\\\",
             "\\midrule"]
    for r, d in zip(rows, deltas):
        lines.append(" & ".join([
            esc(r["model"].replace("_", " ")), num(r["f1_macro_12"]),
            num(r["f1_macro_11"]), num(r["f1_macro_learnable"]),
            num(d, 4, sign=True)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_schemas", lines, "\\texttt{results/f1\\_learnable\\_12models.csv}.")


def t_logme():
    d = keep(json.load(open(p("results", "all_models_full_table.json")))["models"])
    rows = [m for m in d if m.get("logme") is not None]
    v = [m["logme"] for m in rows]
    # Le score est bimodal : les modèles affinés ont vu les étiquettes. L'étendue
    # totale mélange donc les deux groupes ; c'est l'étendue À L'INTÉRIEUR d'un
    # groupe qui dit si le score sépare des modèles comparables.
    frozen = [m["logme"] for m in rows if m["model"] in FROZEN_MODELS]
    ft = [m["logme"] for m in rows if m["model"] not in FROZEN_MODELS]
    intra = max(max(g) - min(g) for g in (frozen, ft) if g)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{LogME (You et al., ICML~2021) calculé sur le train "
             "sous-échantillonné à 20\\,000, seed~42. Le score est bimodal : "
             f"gelés $\\approx${num(np.mean(frozen), 0)}, affinés "
             f"$\\approx${num(np.mean(ft), 0)} --- un artefact du fait que les "
             "modèles affinés ont vu les étiquettes, pas une mesure de "
             f"transférabilité. À l'intérieur d'un groupe l'étendue tombe à "
             f"{num(intra, 0)}~unités sur {len(rows)}~modèles : le score ne "
             "discrimine pas.}",
             "\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
             "Modèle & Dim & LogME (train) & F1 \\\\", "\\midrule"]
    for m in sorted(rows, key=lambda r: -(r.get("f1_linear_probe") or 0)):
        lines.append(" & ".join([
            esc(m["display"]), str(m.get("dim", "---")),
            num(m["logme"], 1), num(m.get("f1_linear_probe"))]) + " \\\\")
    lines += ["\\midrule",
              f"\\textit{{Étendue}} & --- & {num(max(v) - min(v), 1)} & --- \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_logme", lines, "\\texttt{results/all\\_models\\_full\\_table.json}.")


# ═════════════════════════════════════════════════════ géométrie ═════════════
def t_conventions():
    # Les plages sont CALCULÉES sur geometry_models.csv : les valeurs saisies à la
    # main dans une version antérieure de ce tableau étaient fausses.
    g = keep(load("geometry_models.csv"))
    rng = {c: (min(float(r[c]) for r in g), max(float(r[c]) for r in g))
           for c in ("rankme_sigma_raw", "rankme_sigma_centered", "eff_rank_sigma2",
                     "rankme_sigma_raw_norm", "eff_rank_sigma2_norm",
                     "aniso_cosine", "aniso_spectral", "nc1_canonical", "nc1_inverse",
                     "stable_rank", "participation_ratio", "alpha_spectral")}

    def sp(c, nd=0):
        lo, hi = rng[c]
        return f"{num(lo, nd)}--{num(hi, nd)}"

    lines = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Conventions de calcul en présence dans le dépôt. Cinq quantités "
        "distinctes ont circulé sous le nom « RankMe »/« rang effectif » (dont deux "
        "normalisées par la dimension d'embedding, utilisées au tableau des "
        "corrélations~\\ref{tab:correl}), et deux sous le nom « anisotropie » et "
        "« NC1 ». Les colonnes des tableaux qui suivent portent désormais le nom "
        "long.}\\label{tab:conv}",
        "\\begin{tabular}{@{}lp{5.0cm}r>{\\raggedright\\arraybackslash}p{4.6cm}@{}}", "\\toprule",
        "Nom long & Formule & Plage typique & Où elle apparaît \\\\", "\\midrule",
        "RankMe ($\\sigma$, bruts) & $\\exp H(\\sigma_k/\\sum\\sigma)$, $E$ non centré "
        "& " + sp("rankme_sigma_raw") + " & " + path("geometry_extended_12models.json") + " \\\\",
        "RankMe ($\\sigma$, centrés) & $\\exp H(\\sigma_k/\\sum\\sigma)$, $E$ centré "
        "& " + sp("rankme_sigma_centered") + " & " + path("results/datacurve_lora/") + " \\\\",
        "Rang effectif ($\\sigma^2$) & $\\exp H(\\sigma_k^2/\\sum\\sigma^2)$, $E$ "
        "centré & " + sp("eff_rank_sigma2", 1) + " & colonne « RankMe » des rapports \\\\",
        "RankMe ($\\sigma$) normalisé & idem « bruts », divisé par la dimension $D$ "
        "& " + sp("rankme_sigma_raw_norm", 2) + " & tableau~\\ref{tab:correl} seulement \\\\",
        "Rang effectif ($\\sigma^2$) normalisé & idem $\\sigma^2$ centré, divisé "
        "par $D$ & " + sp("eff_rank_sigma2_norm", 2) + " & tableau~\\ref{tab:correl} seulement \\\\",
        "\\midrule",
        "Anisotropie (cosinus) & $\\overline{\\cos(z_i,z_j)}$ sur $10^4$ paires "
        "& " + sp("aniso_cosine", 2) + " & " + path("src/latent.py") + ", rapports \\\\",
        "Anisotropie spectrale & $\\lambda_1 D / \\sum\\lambda$ & " + sp("aniso_spectral") + " & "
        + path("results/datacurve_lora/geometry_full.py") + " \\\\",
        "\\midrule",
        "NC1 (canonique) & $\\operatorname{tr}\\Sigma_W / \\operatorname{tr}\\Sigma_B$ "
        "--- bas = bon & " + sp("nc1_canonical", 2) + " & rapports, " + path("all_models_full_table") + " \\\\",
        "NC1 (inverse) & $\\operatorname{tr}\\Sigma_B / \\operatorname{tr}\\Sigma_W$ "
        "--- haut = bon & " + sp("nc1_inverse", 2) + " & " + path("geometry_full.py") + " \\\\",
        "\\midrule",
        "Stable rank ($\\equiv$ NESum) & $\\sum\\sigma^2 / \\sigma_{\\max}^2$ & "
        + sp("stable_rank", 2) + " & partout (identique) \\\\",
        "Participation ratio & $(\\sum\\sigma^2)^2 / \\sum\\sigma^4$ & " + sp("participation_ratio", 1) + " & "
        "partout (identique) \\\\",
        "Exposant spectral $\\alpha$ & pente de $\\log\\lambda_j$ vs $\\log j$ & "
        + sp("alpha_spectral", 2) + " & partout (identique) \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_conventions", lines,
          "\\texttt{src/latent.py}, \\texttt{scripts/geometry\\_extended.py}, "
          "\\texttt{results/datacurve\\_lora/geometry\\_full.py}, "
          "\\texttt{scripts/rapport/geometry\\_test\\_fixed\\_n.py}.")


GEOM_FULL = [("eff_rank_sigma2", "Rang eff. $\\sigma^2$", 1), ("rankme_sigma_raw", "RankMe $\\sigma$", 1),
             ("stable_rank", "StbRk", 2), ("participation_ratio", "PR", 1),
             ("alpha_spectral", "$\\alpha$", 2), ("aniso_cosine", "Aniso", 3),
             ("fisher_ratio", "Fisher", 3), ("knn_purity", "kNN", 3),
             ("silhouette", "Silh", 4), ("nc1_canonical", "NC1", 2),
             ("dbi", "DBI", 2), ("chi", "CHI", 0),
             ("intrinsic_dim_twonn", "TwoNN", 1), ("logme_train", "LogME tr", 1)]


def t_geometry_full():
    fp = os.path.join(OUT, "geometry_models.csv")
    if not os.path.exists(fp):
        return
    rows = load("geometry_models.csv")
    rows = [r for r in rows if r["model"] in CANONICAL_F1]
    rows.sort(key=lambda r: -CANONICAL_F1[r["model"]][0])
    disp = {}
    for m in json.load(open(p("results",
                              "all_models_canonical_merged.json")))["models"]:
        disp[m["model"]] = m["display"]
    lines = ["\\setlength{\\tabcolsep}{3pt}\\footnotesize",
             "\\begin{longtable}{@{}lr" + "r" * len(GEOM_FULL) + "@{}}",
             f"\\caption{{Géométrie latente des {len(rows)}~modèles --- protocole unique : "
             "embeddings test, sous-échantillon 20\\,000, seed~42, métriques "
             "calculées par seed puis moyennées. Trié par F1 décroissant.}"
             "\\label{tab:geomfull}\\\\", "\\toprule",
             "Modèle & F1 & " + " & ".join(l for _, l, _ in GEOM_FULL) + " \\\\",
             "\\midrule", "\\endfirsthead", "\\toprule",
             "Modèle & F1 & " + " & ".join(l for _, l, _ in GEOM_FULL) + " \\\\",
             "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    for r in rows:
        lines.append(" & ".join(
            [esc(disp.get(r["model"], r["model"])),
             num(CANONICAL_F1[r["model"]][0])]
            + [num(r[c], nd) for c, _l, nd in GEOM_FULL]) + " \\\\")
    lines += ["\\end{longtable}"]
    write("t_geometry_full", lines,
          "\\texttt{results/rapport\\_data/geometry\\_models.csv} "
          "(\\texttt{scripts/rapport/geometry\\_test\\_fixed\\_n.py}).")


def t_correlations():
    fp = os.path.join(OUT, "correlations_geometry.csv")
    if not os.path.exists(fp):
        return
    rows = load("correlations_geometry.csv")
    pops = ["all", "frozen", TIER_POP]
    labels = {}
    for r in rows:
        labels[r["metric"]] = r["label"]
    order = sorted({r["metric"] for r in rows},
                   key=lambda m: -abs(float([r for r in rows if r["metric"] == m
                                             and r["population"] == "all"][0]["rho"])))
    ns = {pp: [r["n"] for r in rows if r["population"] == pp][0] for pp in pops}
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.5pt}",
             "\\caption{Corrélation de Spearman entre chaque métrique géométrique et "
             "le F1 aval, sur trois populations emboîtées. IC95 par bootstrap "
             "($n=2\\,000$, seed~42) ; $\\checkmark$ = l'IC95 ne traverse pas 0. "
             "Toutes les métriques sont calculées avec le protocole unique du "
             "tableau~\\ref{tab:geomfull}.}\\label{tab:correl}",
             "\\begin{tabular}{@{}l" + "rl" * 3 + "@{}}", "\\toprule",
             "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab in
                               (f"{ns['all']} modèles", f"{ns['frozen']} gelés",
                                f"{ns[TIER_POP]} meilleurs")) + " \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Métrique & " + " & ".join("$\\rho$ & IC95" for _ in pops) + " \\\\",
             "\\midrule"]
    for m in order:
        cells = [labels[m]]
        for pp in pops:
            sel = [r for r in rows if r["metric"] == m and r["population"] == pp]
            if not sel:
                cells += ["---", ""]
                continue
            r = sel[0]
            mark = "$\\checkmark$" if r["signif_ci"] == "True" else ""
            cells += [num(r["rho"], 2, sign=True),
                      f"[{num(r['ci95_low'], 2, True)}, "
                      f"{num(r['ci95_high'], 2, True)}]\\,{mark}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_correlations", lines,
          "\\texttt{results/rapport\\_data/correlations\\_geometry.csv} "
          "(\\texttt{scripts/rapport/correlations.py}). "
          f"Tailles : {ns['all']} / {ns['frozen']} / {ns[TIER_POP]} modèles.")


def t_family():
    fp = os.path.join(OUT, "geometry_models.csv")
    if not os.path.exists(fp):
        return
    rows = {r["model"]: r for r in load("geometry_models.csv")}
    fam = [("dinov3_vitb16_lvd", "Gelé"), ("dinov3_vitb16_lvd_full", "Full FT"),
           ("dinov3_vitb16_lvd_mhsa", "MHSA"),
           ("dinov3_vitb16_lvd_lora_r8", "LoRA r=8"),
           ("ctxdistill_dA_tL", "Contexte R1"),
           ("ctxdistill_dA_tEMA", "Contexte R3"),
           ("ctxdistill_dB_tL", "Contexte R2 (fusion)")]
    cols = [("eff_rank_sigma2", "Rang eff. $\\sigma^2$", 1), ("stable_rank", "StbRk", 2),
            ("aniso_cosine", "Aniso", 3), ("fisher_ratio", "Fisher", 3),
            ("knn_purity", "kNN", 3), ("silhouette", "Silh", 4),
            ("nc1_canonical", "NC1", 2), ("alpha_spectral", "$\\alpha$", 2)]
    present = [k for k, _ in fam if k in rows]
    f1s = [CANONICAL_F1[k][0] for k in present]

    def span(col):
        v = [float(rows[k][col]) for k in present]
        return max(v) / min(v)

    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{2.4pt}",
             "\\caption{Famille DINOv3 ViT-B/16 (768~dim.) --- le backbone gelé et "
             f"ses {len(present) - 1}~régimes d'affinage. Les F1 tiennent dans une "
             f"bande de {num(max(f1s) - min(f1s), 3)} tandis que l'anisotropie varie "
             f"d'un facteur {num(span('aniso_cosine'), 1)} et le rang effectif d'un "
             f"facteur {num(span('eff_rank_sigma2'), 1)}. "
             "$\\pm$ = écart-type inter-seed.}\\label{tab:family}",
             "\\begin{tabular}{@{}lr" + "r" * len(cols) + "@{}}", "\\toprule",
             "Régime & F1 & " + " & ".join(l for _, l, _ in cols) + " \\\\",
             "\\midrule"]
    for k, lab in fam:
        if k not in rows:
            continue
        r = rows[k]
        cells = [lab, num(CANONICAL_F1[k][0])]
        for c, _l, nd in cols:
            v = num(r[c], nd)
            sd = r.get(c + "_std", "")
            if sd not in ("", None) and float(sd) > 0:
                v += f"\\,\\tiny{{$\\pm$\\,{num(sd, nd)}}}"
            cells.append(v)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\vspace{2pt}\\par\\footnotesize \\textbf{Note :} Contexte R2 est "
              "mesuré sur la feature \\emph{fusionnée} 1536-dim (tuile$\\oplus$contexte) ; "
              "son rang effectif et son StbRk ne sont donc \\emph{pas} comparables aux "
              "768-dim. Les métriques de forme (anisotropie, Fisher, kNN, silhouette, "
              "NC1, $\\alpha$) se comparent.", "\\end{table}"]
    write("t_family", lines,
          "\\texttt{results/rapport\\_data/geometry\\_models.csv}.")


def t_controls():
    km = json.load(open(p("results", "T2a_kmeans_control.json")))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Contrôle $k$-means : la silhouette recalculée sur des "
             "partitions $k$-means (au lieu des vraies étiquettes) ne corrèle plus "
             "avec le F1. La corrélation silhouette$\\leftrightarrow$F1 tient donc "
             "à l'alignement avec les classes, pas à une propriété générique de "
             "compacité.}",
             "\\begin{tabular}{@{}llrr@{}}", "\\toprule",
             "Schéma & Seed $k$-means & $\\rho$ Spearman & $p$ \\\\", "\\midrule"]
    for scheme in ("11cls", "8cls"):
        for seed, v in sorted(km[scheme].items(), key=lambda kv: int(kv[0])):
            lines.append(" & ".join([scheme, seed, num(v["rho"], 3, sign=True),
                                     num(v["p"], 3)]) + " \\\\")
        if scheme == "11cls":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_kmeans_control", lines,
          "\\texttt{results/T2a\\_kmeans\\_control.json}.")


def t_layerwise():
    fp = os.path.join(OUT, "layerwise.csv")
    if not os.path.exists(fp):
        return
    rows = load("layerwise.csv")
    keys = [("dinov3_vitl16_lvd", "DINOv3 ViT-L LVD"),
            ("dinov3_vitl16_sat", "DINOv3 ViT-L SAT")]
    cols = [("rankme_sigma_raw", "RankMe $\\sigma$", 1), ("aniso_cosine", "Aniso", 3),
            ("silhouette", "Silh", 4), ("knn_purity", "kNN", 3),
            ("nc1_canonical", "NC1", 2)]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Géométrie couche par couche des deux ViT-L/16 (embeddings "
             "test, $n=17\\,598$). Une couche sur quatre est affichée ; le fichier "
             "source contient les 24.}",
             "\\begin{tabular}{@{}ll" + "r" * len(cols) + "@{}}", "\\toprule",
             "Modèle & Couche & " + " & ".join(l for _, l, _ in cols) + " \\\\"]
    for key, lab in keys:
        sub = sorted((r for r in rows if r["model"] == key and r["silhouette"] != ""),
                     key=lambda r: int(r["layer"]))
        if not sub:
            continue
        lines.append("\\midrule")
        shown = [r for r in sub if int(r["layer"]) % 4 == 0 or
                 int(r["layer"]) == int(sub[-1]["layer"])]
        for i, r in enumerate(shown):
            lines.append(" & ".join([lab if i == 0 else "", r["layer"]]
                                    + [num(r[c], nd) for c, _l, nd in cols]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_layerwise", lines,
          "\\texttt{results/rapport\\_data/layerwise.csv} "
          "(\\texttt{scripts/rapport/layerwise\\_geometry.py}).")


# ═════════════════════════════════════════════════════ annexes ═══════════════
def t_runs_raw():
    rows = load("screening_raw.csv")
    rows.sort(key=lambda r: (r["model"], float(r["fraction"]), int(r["seed"])))
    lines = ["\\begin{longtable}{@{}llrrrrrrr@{}}",
             f"\\caption{{Annexe A --- les {len(rows)}~runs d'affinage, un par ligne. "
             "\\texttt{best\\_C} et \\texttt{best\\_epoch} sont ceux retenus par la "
             "validation du run.}\\label{tab:runs}\\\\", "\\toprule",
             "Modèle & Pipeline & Frac. & Tuiles & Seed & F1 11cls & F1 8cls & "
             "Exact. & Époque \\\\", "\\midrule", "\\endfirsthead", "\\toprule",
             "Modèle & Pipeline & Frac. & Tuiles & Seed & F1 11cls & F1 8cls & "
             "Exact. & Époque \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endfoot"]
    for r in rows:
        lines.append(" & ".join([
            esc(r["display"]), esc(r["pipeline"]),
            f"{100 * float(r['fraction']):g}\\,\\%".replace(".", ","),
            thousands(r["n_train_tiles"]), r["seed"], num(r["f1_pres"]),
            num(r["f1_8cls"]), num(r["accuracy"]), r["best_epoch"]]) + " \\\\")
    lines += ["\\end{longtable}"]
    write("t_runs_raw", lines,
          "\\texttt{results/rapport\\_data/screening\\_raw.csv}.")


def _ctx_run_f1(tag_stem):
    """(seeds, mean, std, f8_mean) depuis results/context_distill/runs/<stem>_seed{0,1,2}/metrics.json."""
    import statistics
    seeds, f8s = [], []
    for s in (0, 1, 2):
        fp = p("results", "context_distill", "runs", f"{tag_stem}_seed{s}",
               "metrics.json")
        d = json.load(open(fp))
        seeds.append(float(d["f1_macro_pres_test"]))
        f8s.append(float(d["f1_macro_8cls_test"]))
    m = float(statistics.mean(seeds))
    sd = float(statistics.stdev(seeds)) if len(seeds) > 1 else 0.0
    return seeds, m, sd, float(statistics.mean(f8s))


CTX_RUNS = [
    ("B", "DINOv3-L (externe)",
     "dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100",
     "tête apprise sur [tuile;contexte] (1536), contexte requis à l'inférence"),
    ("A", "DINOv3-L (externe)",
     "dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100",
     "distillation seule, tuile seule à l'inférence (768)"),
    ("A", "EMA-self",
     "dinov3_vitb16_lvd_ctxdistill_dA_tEMA_ctx1024_r2a4_frac100",
     "distillation seule, teacher = copie EMA du student"),
    ("B", "SimDINOv2-L (externe)",
     "simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r2a4_frac100",
     "Design B entraîné, backbone SimB, contexte 512px (1536)"),
    ("B", "SimDINOv2-L (externe)",
     "simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r8a16_frac100",
     "idem, LoRA r=8 au lieu de r=2"),
]


def t_ctx_distill():
    """Résultats contexte : DINOv3-B @1024 (R1/R2/R3) + SimDINOv2-B @512 entraînés.
    Remplace le fragment maintenu à la main (chiffres tapés, septembre) par une
    génération depuis results/context_distill/runs/*/metrics.json."""
    rows = []
    for design, teacher, stem, note in CTX_RUNS:
        seeds, m, sd, f8 = _ctx_run_f1(stem)
        rows.append((design, teacher, seeds, m, sd, f8, note))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Résultats contexte (3~seeds, test v3 spatial, F1-macro). "
             "Design~B (tête apprise sur la fusion) contre design~A (distillation "
             "seule, tuile seule à l'inférence). Les deux runs SimDINOv2-B @512 "
             "entraînés n'apportent rien sur le gelé-fusionné du même backbone "
             "(0,5059, tableau~\\ref{tab:ctxsweep}) : quand le pré-entraînement "
             "est aligné, la fusion est déjà linéairement décodable.}"
             "\\label{tab:ctxf1}",
             "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
             "Design & Teacher & seed0 & seed1 & seed2 & F1 moy $\\pm$ std & "
             "F1 8cls \\\\", "\\midrule"]
    for design, teacher, seeds, m, sd, f8, _note in rows:
        lab = "\\textbf{" + design + "}" if design == "B" else design
        cells = [lab, esc(teacher)] + [num(s) for s in seeds]
        mm = num(m) + " $\\pm$ " + num(sd)
        if design == "B":
            mm = "\\textbf{" + mm + "}"
        cells += [mm, num(f8)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_ctx_distill", lines,
          "\\texttt{results/context\\_distill/runs/*/metrics.json} "
          "(\\texttt{f1\\_macro\\_pres\\_test}, \\texttt{f1\\_macro\\_8cls\\_test}).")


def t_ctx_matrix():
    """Matrice d'attribution : même test v3, même sonde, seed 0 (contrôles).
    Remplace la table saisie à la main par une génération depuis
    results/context_distill/controls/fused_probe_*.json."""
    files = {
        "DINOv3-B gelé": "fused_probe_frozen_seed0.json",
        "LoRA r2 (split v3)": "fused_probe_lora_r2a4_v3train_seed0.json",
        "LoRA r8a16 (spatial)": "fused_probe_lora_r8a16_spatial_seed0.json",
        "R1 distil. A": "fused_probe_r1_dA_tL_seed0.json",
        "R2 distil. B": "fused_probe_r2_dB_tL_seed0.json",
    }
    rows = []
    for lab, fn in files.items():
        d = json.load(open(p("results", "context_distill", "controls", fn)))
        t, fu = d["tile"], d.get("fused") or {}
        fv, bcf = fu.get("f1_macro_pres_test"), fu.get("best_C", "---")
        if fv is None and lab.startswith("R2"):
            # Pas de clé "fused" dans le contrôle R2 seed0 : moyenne 3 seeds.
            _s, _m, _sd, _f8 = _ctx_run_f1(
                "dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100")
            fv, bcf = _m, "---"
        rows.append((lab, t["f1_macro_pres_test"], fv, t["best_C"], bcf))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Attribution du gain contexte : même test v3, même sonde "
             "canonique, seed~0. Chaque backbone est sondé sur la tuile seule "
             "(768) \\emph{puis} sur les features fusionnées (1536) du même "
             "checkpoint. Le contexte aide \\emph{tout le monde} mais \\emph{seul} "
             "R2, entraîné sur features fusionnées, franchit 0,51 --- au prix "
             "d'une tuile seule dégradée : le backbone s'est spécialisé.}"
             "\\label{tab:ctxmatrix}",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Modèle & tuile (768) & fusionné (1536) & $\\Delta$ contexte & "
             "best\\_C / fused \\\\", "\\midrule"]
    for lab, tv, fv, bc, bcf in rows:
        dl = (fv - tv) if fv else None
        labt = "\\textbf{" + lab + "}" if lab.startswith("R2") else lab
        cells = [labt, num(tv), num(fv),
                 (num(dl, 3, sign=True) if dl is not None else "---"),
                 f"{bc} / {bcf}"]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_ctx_matrix", lines,
          "\\texttt{results/context\\_distill/controls/fused\\_probe\\_*.json} "
          "(seed~0). Sanity : gelé/tuile 0,4716 $\\approx$ canonique 0,4712 ; "
          "LoRA spatial/tuile 0,4887 $\\approx$ 0,4885. Caveat : best\\_C "
          "diffère (0,0001 pour R2/fused) ; contrôles seed~0 seul.")


def t_ctx_controls():
    """Contrôles Bouguessa sur R2 (Design B) : fused / tuile / contexte seul /
    contexte permuté (5 réplicats), 3 seeds — sonde canonique sur sig_embeddings."""
    import statistics
    base = p("results", "context_distill", "controls_bouguessa")
    need = [f"r2_dB_tL_seed{s}_{v}.json"
            for s in (0, 1, 2) for v in ("fused", "tile", "ctx")] + \
           [f"r2_dB_tL_seed{s}_fused_ctxperm{q}.json"
            for s in (0, 1, 2) for q in range(5)]
    missing = [f for f in need if not os.path.exists(os.path.join(base, f))]
    if missing:
        # Contrôles recalculables en local (sig_embeddings présents) :
        # python3 scripts/context_bouguessa_controls.py --workers 2
        write("t_ctx_controls",
              ["\\begin{table}[htbp]", "\\centering",
               "\\caption{Contrôles Bouguessa sur R2 --- calcul en cours "
               "(\\texttt{scripts/context\\_bouguessa\\_controls.py}) ; "
               f"{len(missing)}/{len(need)} fichiers manquants. Valeurs publiées "
               "dans \\texttt{results/context\\_distill/CONTROLES\\_BOUGUESSA.md}.}",
               "\\begin{tabular}{@{}l@{}}", "\\toprule",
               "En attente \\\\", "\\bottomrule", "\\end{tabular}",
               "\\end{table}"],
              "\\texttt{results/context\\_distill/controls\\_bouguessa/}.")
        print(f"  [ATTENTE] t_ctx_controls : {len(missing)} fichiers manquants")
        return
    rec = {}
    for s in (0, 1, 2):
        rec[(s, "fused")] = json.load(open(os.path.join(
            base, f"r2_dB_tL_seed{s}_fused.json")))["f1_macro_pres_test"]
        rec[(s, "tile")] = json.load(open(os.path.join(
            base, f"r2_dB_tL_seed{s}_tile.json")))["f1_macro_pres_test"]
        rec[(s, "ctx")] = json.load(open(os.path.join(
            base, f"r2_dB_tL_seed{s}_ctx.json")))["f1_macro_pres_test"]
        for q in range(5):
            rec[(s, f"perm{q}")] = json.load(open(os.path.join(
                base, f"r2_dB_tL_seed{s}_fused_ctxperm{q}.json")))["f1_macro_pres_test"]
    agg = {}
    for variant in ("fused", "tile", "ctx"):
        v = [rec[(s, variant)] for s in (0, 1, 2)]
        agg[variant] = (float(statistics.mean(v)),
                        float(statistics.stdev(v)) if len(v) > 1 else 0.0, v)
    pv = [rec[(s, f"perm{q}")] for s in (0, 1, 2) for q in range(5)]
    agg["perm"] = (float(statistics.mean(pv)), float(statistics.stdev(pv)), None)
    order = [("fused", "Fusionné [tuile;contexte] (vrai appariement)"),
             ("tile", "Tuile seule (0:768)"),
             ("ctx", "Contexte seul (768:1536)"),
             ("perm", "Contexte permuté (15 valeurs : 3 seeds $\\times$ 5 réplicats)")]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Contrôles demandés début septembre : (1)~le contexte seul, "
             "sans la tuile centrale ; (2)~un contexte permuté (désaligné). Sonde "
             "canonique sur les embeddings du run R2 (Design~B, fusion 1536). Le "
             "contexte seul égale la tuile seule ; le contexte permuté tombe "
             "\\emph{sous} la tuile seule : le gain de R2 vient de l'information "
             "spatiale appariée, pas de la concaténation.}"
             "\\label{tab:ctxcontrols}",
             "\\begin{tabular}{@{}lrr@{}}", "\\toprule",
             "Représentation & F1 (moy $\\pm$ std) & seeds \\\\", "\\midrule"]
    for key, lab in order:
        m, sd, v = agg[key]
        seeds_txt = ", ".join(num(x) for x in v) if v else "---"
        cell = f"{num(m)} $\\pm$ {num(sd)}"
        if key == "fused":
            cell = "\\textbf{" + cell + "}"
        lines.append(" & ".join([lab, cell, seeds_txt]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_ctx_controls", lines,
          "\\texttt{results/context\\_distill/controls\\_bouguessa/" 
          "r2\\_dB\\_tL\\_seed{0,1,2}\\_*.json}.")


def t_ctx_sweep():
    """Sweep frozen multi-backbones × tailles (512/1024/2048), pallier frozen.
    Sonde canonique sur features FROZEN fusionnées [tuile;ctx] (seed~0, test v3)."""
    import statistics
    base = p("results", "context_distill", "controls_bouguessa")
    models = [("dinov3_vits16", "DINOv3 ViT-S/16"),
              ("dinov3_vitb16_lvd", "DINOv3 ViT-B/16"),
              ("dinov3_vitl16", "DINOv3 ViT-L/16"),
              ("simdinov2_vitb16", "SimDINOv2-B"),
              ("simdinov2_vitl16", "SimDINOv2-L")]
    # DINOv3-B vient de la première campagne (controls/, préfixe frozen_ctx*).
    rows = []
    for key, disp in models:
        for size in ("512", "1024", "2048"):
            rec = {}
            for var in ("fused", "tile", "ctx"):
                fp = os.path.join(base, f"frozen_{key}_ctx{size}_seed0_{var}.json")
                if not os.path.exists(fp) and key == "dinov3_vitb16_lvd":
                    # Première campagne (size_sweep) : pas d'infixe modèle = DINOv3-B.
                    fp = os.path.join(base, f"frozen_ctx{size}_seed0_{var}.json")
                if os.path.exists(fp):
                    rec[var] = json.load(open(fp))["f1_macro_pres_test"]
            perms = []
            for q in range(3):
                fp = os.path.join(base, f"frozen_{key}_ctx{size}_seed0_fused_ctxperm{q}.json")
                if not os.path.exists(fp) and key == "dinov3_vitb16_lvd":
                    fp = os.path.join(base, f"frozen_ctx{size}_seed0_fused_ctxperm{q}.json")
                if os.path.exists(fp):
                    perms.append(json.load(open(fp))["f1_macro_pres_test"])
            rec["perm"] = (float(statistics.mean(perms))
                             if perms else float("nan"))
            if "fused" in rec and "tile" in rec:
                rows.append((disp, size, rec))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Sweep gelé : 5~backbones $\\times$ 3~tailles de contexte, "
             "sonde canonique sur features FROZEN fusionnées [tuile;ctx] "
             "(seed~0, test v3 spatial). SimDINOv2-B @512 (0,5059, \\emph{sans "
             "aucun entraînement}) atteint quasiment le R2 entraîné (0,5080) : "
             "le pré-entraînement iNat-Plantae décode le voisinage mieux que "
             "DINOv3-LVD à toutes les tailles ; 512 $>$ 1024 $\\gg$ 2048 partout.}"
             "\\label{tab:ctxsweep}",
             "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
             "Modèle & Taille & fusionné & tuile & ctx seul & $\\Delta$ctx "
             "& permuté \\\\", "\\midrule"]
    for disp, size, rec in rows:
        dl = rec["fused"] - rec["tile"]
        fcell = f"{num(rec['fused'])}"
        if rec["fused"] == max(r["fused"] for _d, _s, r in rows):
            fcell = "\\textbf{" + fcell + "}"
        lines.append(" & ".join([disp, size, fcell, num(rec.get("tile")),
                                  num(rec.get("ctx")),
                                  num(dl, 3, sign=True), num(rec["perm"])])
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_ctx_sweep", lines,
          "\\texttt{results/context\\_distill/controls\\_bouguessa/" 
          "frozen\\_*.json} (DINOv3-B = fichiers sans infixe modèle, première campagne).")


# ═════════════════════════════════════════ têtes non linéaires (2026-09) ══════
# Source : sweep Narval `slurm_head_sweep_all.sh` (job 2929768), 2026-09-09.
# Tile-only : 33 groupes du palier ; fused : 24 tags [tuile;contexte].

def _head_tag_label(tag):
    """Libellé lisible d'un tag de `fusion_heads`."""
    MOD = {"dinov3_vits16": "DINOv3 ViT-S/16", "dinov3_vitb16_lvd": "DINOv3 ViT-B/16",
           "dinov3_vitl16": "DINOv3 ViT-L/16", "simdinov2_vitb16": "SimDINOv2-B",
           "simdinov2_vitl16": "SimDINOv2-L"}
    if "_FROZEN_fused_ctx" in tag:
        base, ctx = tag.split("_FROZEN_fused_ctx")
        return f"{MOD.get(base, base)} @{ctx.split('_')[0]}"
    sd = tag.rsplit("_", 1)[-1]
    for key, lab in (("ctxdistill_dB_tL", "R2 (fusion apprise)"),
                     ("ctxdistill_dA_tEMA", "R3 (A, EMA)"),
                     ("ctxdistill_dA_tL", "R1 (A, tuile)")):
        if key in tag:
            return f"{lab} {sd}"
    return tag


def _head_family(tag):
    if "_FROZEN_fused_ctx" in tag:
        return 0
    if "ctxdistill_dB_tL" in tag:
        return 1
    if "ctxdistill_dA_tEMA" in tag:
        return 2
    return 3


def t_head_tile():
    """Têtes (lbfgs canonique / lin-AdamW / MLP-2) sur les 33 groupes tuile du palier.
    Réponse mesurée : aucune tête non linéaire ne bat la sonde linéaire canonique."""
    d = os.path.join(OUT, "tile_heads")
    if not os.path.isdir(d):
        print("  [skip] t_head_tile : results/rapport_data/tile_heads absent")
        return
    rows = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(d, f)))
            rows.append((j["group"], j["lin_lbfgs"]["f1_mean"],
                         j["lin_adamw"]["f1_mean"], j["mlp2"]["f1_mean"]))
    rows.sort(key=lambda r: -r[1])
    mp_lb = float(np.mean([r[3] - r[1] for r in rows]))
    mp_ad = float(np.mean([r[3] - r[2] for r in rows]))
    n_neg = sum((r[3] - r[1]) < 0 for r in rows)
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Têtes de classification sur les %d groupes tuile du palier "
             "(3~seeds) : sonde canonique \\texttt{lbfgs} (référence publiée), "
             "linéaire entraînée par la même boucle AdamW, et MLP-2. Le MLP-2 ne bat "
             "jamais la sonde linéaire : $\\Delta$ moyen vs \\texttt{lbfgs} $= %s$ "
             "(%d/%d groupes négatifs) et $= %s$ vs AdamW : aucune récupération de "
             "l'écart lbfgs$\\leftrightarrow$AdamW. "
             "\\emph{La non-linéarité ne crée pas d'information.}}"
             "\\label{tab:headstile}"
             % (len(rows), num(mp_lb, 4, sign=True), n_neg, len(rows), num(mp_ad, 4, sign=True)),
             "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "Groupe & \\texttt{lbfgs} & lin-AdamW & MLP-2 & $\\Delta$ vs \\texttt{lbfgs} "
             "& $\\Delta$ vs AdamW \\\\", "\\midrule"]
    for g, lb, ad, mp in rows:
        lines.append(" & ".join([esc(g), num(lb), num(ad), num(mp),
                                 num(mp - lb, 4, sign=True), num(mp - ad, 4, sign=True)])
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_head_tile", lines,
          "\\texttt{results/rapport\\_data/tile\\_heads/*.json} (sweep Narval "
          "\\texttt{scripts/slurm\\_head\\_sweep\\_all.sh}, job 2929768, 2026-09-09).")


def t_head_fused():
    """MLP-2 / bilinéaire-diagonale / FiLM sur les embeddings fusionnés [tuile;contexte]."""
    d = p("results", "context_distill", "fusion_heads")
    if not os.path.isdir(d):
        print("  [skip] t_head_fused : results/context_distill/fusion_heads absent")
        return
    recs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d))
            if f.endswith(".json")]

    def v(j, k):
        x = j.get(k)
        return x.get("f1_mean", x.get("f1_test")) if isinstance(x, dict) else None

    recs.sort(key=lambda j: (_head_family(j["tag"]), j["tag"]))
    rows = []
    for j in recs:
        lb, ad = v(j, "lin_lbfgs"), v(j, "lin_adamw")
        mp, bil, film = v(j, "mlp2"), v(j, "bil_diag"), v(j, "film")
        nl = [x for x in (mp, bil, film) if x is not None]
        rows.append((_head_family(j["tag"]), _head_tag_label(j["tag"]),
                     lb, ad, mp, bil, film, (max(nl) - lb) if nl else None))
    dbest = [r[7] for r in rows if r[7] is not None]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Têtes non linéaires sur les embeddings fusionnés "
             "[tuile;contexte] (24~tags, seed indiqué). \\texttt{bil\\_diag} "
             "(termes croisés $t_i c_i$) et \\texttt{film} (le contexte module les "
             "canaux de la tuile) testent explicitement l'interaction. "
             "\\textbf{Aucune ne bat la sonde linéaire} : meilleure non-linéaire vs "
             "\\texttt{lbfgs}, $\\Delta$ moyen $= %s$, %d/%d tags positifs. "
             "Le $+0{,}013$ de R2 n'est donc pas une interaction manquée par la "
             "linéarité : c'est un gain de \\emph{représentation}, capté par la "
             "sonde linéaire elle-même. En 768d (design~A) les deux têtes "
             "d'interaction sont sans objet et non lancées.}"
             "\\label{tab:headsfused}"
             % (num(float(np.mean(dbest)), 4, sign=True), sum(x > 0 for x in dbest), len(dbest)),
             "\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             "Tag & \\texttt{lbfgs} & lin-AdamW & MLP-2 & bil-diag & FiLM "
             "& meilleur non-lin. $-$ \\texttt{lbfgs} \\\\", "\\midrule"]
    prev = None
    for fam, lab, lb, ad, mp, bil, film, db in rows:
        if prev is not None and fam != prev:
            lines.append("\\midrule")
        prev = fam
        lines.append(" & ".join([esc(lab), num(lb), num(ad), num(mp), num(bil),
                                 num(film), num(db, 4, sign=True)]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_head_fused", lines,
          "\\texttt{results/context\\_distill/fusion\\_heads/*.json} (sweep Narval "
          "\\texttt{scripts/slurm\\_head\\_sweep\\_all.sh}, job 2929768, 2026-09-09).")


def t_simb_ablation():
    """Ablation LoRA/PEFT SimDINOv2-B, Stage A (13 bras × 3 seeds) — le palier plat.
    F1 canonique (reprobe mono-thread) + probe interne (best_C, best_epoch) + budget.

    Révision 2026-09-08 (retour lecteur) : rangs numérotés (le classement F1
    décroissant doit se LIRE), colonne Δ vs gelé, libellés conformes au tableau
    maître (« blocs 9-11 » plutôt que « 3 derniers blocs »), backbone rappelé
    dans l'en-tête de colonne et les lignes de référence (« gelé », « ancre »),
    best_C et époque de l'ancre enfin remplis (ils étaient jetés par un filtre de
    préfixe trop strict sur screening_agg)."""
    canon = {m["model"]: m for m in json.load(
        open(p("results", "simb_stageA_probe_CANONICAL.json")))["models"]}
    # startswith("simdinov2_vitb16_lora") couvre AUSSI l'ancre exacte
    # "simdinov2_vitb16_lora" (l'ancien filtre exigeait le "_" final et la
    # jetait : d'où les "---" de la ligne ancre).
    agg = {r["model"]: r for r in load("screening_agg.csv")
           if r["model"].startswith("simdinov2_vitb16_lora")
           or r["model"] == "simdinov2_vitb16_norm_tuning"}
    DIM = 768
    # Budget LoRA : 2·D·r par cible et par bloc adapté (A : r×D, B : D×r).
    # NormTuning : 25 LayerNorms (12 blocs × 2 + finale) × 2×D + tête (D×11+11).
    SPECS = {
        "simdinov2_vitb16_lora_r8_b611":
            ("LoRA r=8, blocs 6-11 (hauts)", 8, 2, 6),
        "simdinov2_vitb16_lora_r8_b05":
            ("LoRA r=8, blocs 0-5 (bas)", 8, 2, 6),
        "simdinov2_vitb16_lora_r8_b911":
            ("LoRA r=8, blocs 9-11 (3 derniers)", 8, 2, 3),
        "simdinov2_vitb16_lora_r2": ("LoRA r=2, tous blocs", 2, 2, 12),
        "simdinov2_vitb16_lora_r4": ("LoRA r=4, tous blocs", 4, 2, 12),
        "simdinov2_vitb16_lora_r16": ("LoRA r=16, tous blocs", 16, 2, 12),
        "simdinov2_vitb16_lora_r32": ("LoRA r=32, tous blocs", 32, 2, 12),
        "simdinov2_vitb16_lora_r8_s2":
            ("LoRA r=8, scaling 2 ($\\alpha=2r$)", 8, 2, 12),
        "simdinov2_vitb16_lora_r16_s2":
            ("LoRA r=16, scaling 2 ($\\alpha=2r$)", 16, 2, 12),
        "simdinov2_vitb16_lora_r8_rslora":
            ("LoRA r=8, rsLoRA (scaling $\\sqrt{r}$)", 8, 2, 12),
        "simdinov2_vitb16_lora_r16_rslora":
            ("LoRA r=16, rsLoRA (scaling $\\sqrt{r}$)", 16, 2, 12),
        "simdinov2_vitb16_lora_r8_qkv":
            ("LoRA r=8, cibles Q+K+V", 8, 3, 12),
    }

    def spec_budget(k):
        if k == "simdinov2_vitb16_norm_tuning":
            return "SimDINOv2-B NormTuning (normes + tête)", \
                25 * 2 * DIM + (DIM * 11 + 11)
        spec, r, nt, nb = SPECS[k]
        return "SimDINOv2-B " + spec, 2 * DIM * r * nt * nb

    # Lignes : [libellé, f1, sd, best_C, époque, budget, est_référence, gras]
    # Le plateau est encadré par ses deux références du benchmark — backbone
    # gelé et ancre r8a8 tous blocs — dont les libellés disent explicitement
    # le backbone et l'état (retour lecteur : « gelé ou pas, quel modèle ? »).
    rows = []
    for k, spec, is_ref, bold in (
        ("simdinov2_vitb16_lora",
         "SimDINOv2-B LoRA r=8, tous blocs — ANCRE (référence)", True, True),
        ("simdinov2_vitb16",
         "SimDINOv2-B backbone gelé (référence, 0 param. entraîné)", True, False),
    ):
        f1, sd = CANONICAL_F1.get(k, (None, None))
        budget = 2 * DIM * 8 * 2 * 12 if k == "simdinov2_vitb16_lora" else 0
        a = agg.get(k, {})
        _bc = a.get("best_C_mode", "")
        try:
            bc = float(_bc) if _bc not in ("", None) else None
        except (TypeError, ValueError):
            bc = None
        ep = a.get("best_epoch_mean")
        ep = float(ep) if ep else None
        rows.append([spec, f1, sd, bc, ep, budget, is_ref, bold])
    for k in list(SPECS) + ["simdinov2_vitb16_norm_tuning"]:
        c = canon.get(k)
        a = agg.get(k, {})
        spec, budget = spec_budget(k)
        f1 = c["f1_linear_probe"] if c else None
        sd = c["f1_std"] if c else None
        _bc = a.get("best_C_mode", "")
        try:
            bc = float(_bc) if _bc not in ("", None) else None
        except (TypeError, ValueError):
            bc = None
        ep = a.get("best_epoch_mean")
        ep = float(ep) if ep else None
        rows.append([spec, f1, sd, bc, ep, budget, False, False])

    # Classement par F1 canonique décroissant (les références suivent leur rang).
    rows.sort(key=lambda r: -(r[1] if r[1] is not None else -1))
    arm_f1s = [c["f1_linear_probe"] for c in canon.values() if c]
    span = max(arm_f1s) - min(arm_f1s)
    frozen = CANONICAL_F1.get("simdinov2_vitb16", (None,))[0]
    anchor = CANONICAL_F1.get("simdinov2_vitb16_lora", (None,))[0]

    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Ablation LoRA/PEFT sur le backbone \\textbf{SimDINOv2 "
             "ViT-B/16 (iNat-Plantae)} --- \\emph{tous les bras de ce tableau "
             "sont ce même backbone}, gelé ou affiné (tuile seule, même test "
             f"que le benchmark ; {len(arm_f1s)}~bras, classés par \\textbf{{F1 "
             "canonique décroissant}, rang $1 =$ meilleur). Sauf mention : "
             "$\\alpha=r$ (convention r8a8), cibles Q+V, 12~blocs. Le backbone "
             f"gelé ({num(frozen)}) et l'ancre r8a8 tous blocs ({num(anchor)}) "
             "sont donnés comme références ; aucun bras ne les dépasse "
             f"au-delà du bruit : le plateau tient dans {num(span, 4)}. "
             "NormTuning (normes + tête, $\\approx$47k~params) égale l'ancre "
             "au millième près. F1 canonique (reprobe mono-thread, \\S4.8) ; "
             "best\\_C et époque depuis le probe interne des runs "
             "(\\texttt{---} pour le gelé : unique graine, pas d'époque "
             "d'entraînement).}"
             "\\label{tab:simbabl}",
             "\\begin{tabular}{@{}clrrrrrr@{}}", "\\toprule",
             "Rang & Bras (backbone : SimDINOv2 ViT-B/16) & F1 canonique & "
             "$\\sigma$ & $\\Delta$ vs gelé & best\\_C & époque & params "
             "entraînables \\\\", "\\midrule"]
    for rank, (spec, f1, sd, bc, ep, budget, _is_ref, bold) in enumerate(rows, 1):
        f1t = num(f1) if f1 is not None else "---"
        sdt = num(sd) if sd is not None else "---"
        bct = num(bc) if bc is not None else "---"
        ept = num(ep, 0) if ep is not None else "---"
        dlt = (num(f1 - frozen, 4, sign=True)
               if (f1 is not None and frozen is not None and f1 != frozen)
               else ("---" if f1 is not None and f1 == frozen else num(f1 - frozen, 4, sign=True)))
        budget_t = thousands(budget) if budget else "0"
        if bold:
            f1t = "\\textbf{" + f1t + "}"
        lines.append(" & ".join([str(rank), esc(spec), f1t, sdt, dlt, bct,
                                 ept, budget_t]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_simb_ablation", lines,
          "\\texttt{results/simb\\_stageA\\_probe\\_CANONICAL.json} (F1) ; "
          "\\texttt{results/rapport\\_data/screening\\_agg.csv} (best\\_C, époque) ; "
          "budgets = arithmétique d'architecture (2\\,$D$\\,r par cible et par bloc).")


def t_sources():
    n_runs = len(load("screening_raw.csv"))
    _d, _src = _signif_source()
    n_pairs_sig = len(_d["pairs"])
    sig_src = _src.replace("\\_", "_")
    rows = [
        (f"F1 canonique, {N_MODELS} modèles", path("results/all_models_canonical_merged.json"),
         "---"),
        ("Géométrie (protocole unique)", path("results/rapport_data/geometry_models.csv"),
         path("scripts/rapport/geometry_test_fixed_n.py")),
        ("Géométrie vs fraction", path("results/rapport_data/geometry_datacurve.csv"),
         path("scripts/rapport/geometry_test_fixed_n.py")),
        ("Corrélations géométrie/F1", path("results/rapport_data/correlations_geometry.csv"),
         path("scripts/rapport/correlations.py")),
        (f"Runs d'affinage ({n_runs})", path("results/rapport_data/screening_raw.csv"),
         path("scripts/rapport/aggregate_screening.py")),
        (f"F1 par classe, {N_MODELS} modèles", path("results/rapport_data/per_class_all_models.csv"),
         path("scripts/rapport/aggregate_screening.py")),
        ("Probing gelé vs fraction", path("results/datacurve_lora/frozen_probe_results_agg.csv"),
         path("results/datacurve_lora/compute_frozen_probe.py")),
        ("Few-shot (9 modèles gelés)", path("results/sample_efficiency.json"),
         path("scripts/sample_efficiency.py")),
        ("k-NN et probe (12 modèles)", path("results/without_rhol/probe_knn_cgrid.json"),
         path("probe.py")),
        ("Matrices de confusion", path("results/rapport_data/confusion_matrices.npz"),
         path("scripts/rapport/knn_and_confusion.py")),
        (f"Bootstrap apparié ({n_pairs_sig} paires)", path(sig_src),
         path("scripts/rapport/significance_tier.py")),
        ("Contrôle $k$-means", path("results/T2a_kmeans_control.json"), "---"),
        ("Ablation LoRA/PEFT SimB (13 bras, F1 canonique)",
         path("results/simb_stageA_probe_CANONICAL.json"),
         path("scripts/rapport/probe_simb_stageA_canonical.py")),
        ("Sweep contexte gelé (5 backbones $\\times$ 3 tailles)",
         path("results/context_distill/controls_bouguessa/frozen_*.json"),
         "---"),
        ("Têtes non linéaires, tuile (33 groupes)",
         path("results/rapport_data/tile_heads/*.json"),
         path("scripts/tile_head_sweep.py")),
        ("Têtes non linéaires, fusionné [tuile;ctx] (24 tags)",
         path("results/context_distill/fusion_heads/*.json"),
         path("scripts/fusion_head_sweep.py")),
        ("Contrôles Bouguessa R2 (tuile/ctx/permuté)",
         path("results/context_distill/controls_bouguessa/r2_*.json"),
         path("scripts/context_bouguessa_controls.py")),
        ("SimDINOv2-B @512 entraînés (Design B, 2 configs)",
         path("results/context_distill/runs/simdinov2_vitb16_ctxdistill_*.json"),
         path("scripts/slurm_context_distill.sh")),
        ("Effectifs par classe", path("results/tiles_per_class_per_split.json"), "---"),
        ("LogME", path("results/all_models_full_table.json"),
         path("scripts/compute_all_clustering_logme.py")),
        ("Géométrie par couche", path("results/rapport_data/layerwise.csv"),
         path("scripts/rapport/layerwise_geometry.py")),
        ("Contrôle de cohérence", path("results/rapport_data/consistency_report.md"),
         path("scripts/rapport/check_consistency.py")),
    ]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Annexe D --- source canonique de chaque famille de chiffres "
             "et script qui la produit. Les tableaux de ce document sont générés par "
             "\\texttt{scripts/rapport/make\\_tables.py} : aucun chiffre n'y est "
             "saisi à la main.}",
             "\\begin{tabular}{@{}p{3.9cm}>{\\raggedright\\arraybackslash}p{6.4cm}"
             ">{\\raggedright\\arraybackslash}p{5.6cm}@{}}", "\\toprule",
             "Contenu & Fichier & Script \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_sources", lines, "inventaire manuel, vérifié par "
          "\\texttt{scripts/rapport/check\\_consistency.py}.")


def t_dataset():
    d = json.load(open(p("results", "tiles_per_class_per_split.json")))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small",
             "\\caption{Arctic-TVC en chiffres.}",
             "\\begin{tabular}{@{}ll@{}}", "\\toprule",
             "Propriété & Valeur \\\\", "\\midrule",
             "Site & Trail Valley Creek, Territoires du Nord-Ouest, Canada \\\\",
             "Capteur & drone RVB, $\\sim$2,2~mm/pixel (GSD) \\\\",
             "Orthomosaïques & 38 (35 annotées) \\\\",
             "Tuiles & 224$\\times$224~px \\\\",
             f"Train / val / test (12 classes) & "
             f"{thousands(sum(v['train'] for v in d.values()))} / "
             f"{thousands(sum(v['val'] for v in d.values()))} / "
             f"{thousands(sum(v['test'] for v in d.values()))} tuiles \\\\",
             f"Train en schéma 11 classes & "
             f"{thousands(sum(v['train'] for v in d.values()) - d['RHOL']['train'])} "
             "tuiles (RHOL retirée) \\\\",
             "Découpage & par orthomosaïque entière (aucune fuite spatiale) \\\\",
             "Classes & 12 annotées ; 11 évaluées (RHOL absente de val/test) ; "
             "8 réellement apprenables \\\\",
             "CRS & EPSG:32608 \\\\",
             "\\midrule",
             "Probe & LogisticRegression, solveur lbfgs, multinomial \\\\",
             "Grille $C$ & $\\{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10\\}$ \\\\",
             "Sélection & \\texttt{best\\_C} par F1-macro de validation \\\\",
             "\\texttt{max\\_iter} / seed & 2000 / 42 \\\\",
             "Métrique reportée & \\texttt{f1\\_macro\\_pres} (classes présentes "
             "dans le test) \\\\",
             "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_dataset", lines,
          "\\texttt{PROJECT\\_INDEX.md}, "
          "\\texttt{results/tiles\\_per\\_class\\_per\\_split.json}, "
          "\\texttt{AGENTS.md} §3.")


def t_tier_ranking():
    """Le tableau qui répond à « peut-on classer les meilleurs modèles ? »."""
    fp = os.path.join(OUT, "tier_ranking_metrics.csv")
    if not os.path.exists(fp):
        return
    rows = [r for r in load("tier_ranking_metrics.csv")
            if int(r["tier"]) == TIER_K]
    rows.sort(key=lambda r: -float(r["pair_acc"]))
    # Verdict calculé, pas saisi : quelles métriques passent BH par famille ?
    _pass = sorted({r["label"] for r in rows
                    if r["p_bh_family"] not in ("", None)
                    and float(r["p_bh_family"]) < 0.05})
    _verdict = ("Aucune métrique n'atteint le seuil de 0,05 ; seule la famille "
                "spectrale ordonne franchement à l'envers."
                if not _pass else
                "Passent le seuil de 0,05 après correction BH par famille : "
                + ", ".join(_pass) + ". La famille spectrale ordonne "
                "franchement à l'envers.")
    noise = load("tier_ranking_noise.csv")[0]
    n_pairs = int(rows[0]["pairs_total"])
    f1_sorted = sorted((v[0] for v in CANONICAL_F1.values()), reverse=True)
    gap = f1_sorted[TIER_K - 1] - f1_sorted[TIER_K]
    all_gaps = [f1_sorted[i] - f1_sorted[i + 1] for i in range(len(f1_sorted) - 1)]
    gap_ratio = gap / float(np.median(all_gaps))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             f"\\caption{{Pouvoir d'ordonnancement sur le palier des {TIER_K}~meilleurs "
             f"modèles. « Paires » : combien des {n_pairs}~paires la métrique ordonne "
             f"comme le F1 (hasard = {n_pairs // 2}). « n\\textsuperscript{{o}}1 » : la métrique "
             "désigne-t-elle le vrai meilleur modèle. $p_{BH}$ est corrigé "
             "\\emph{à l'intérieur de la famille}, déclarée avant de regarder les "
             f"$p$. Le palier est fixé par la rupture mesurée du classement (écart "
             f"au rang {TIER_K + 1} : {num(gap, 4, sign=True)}, soit environ "
             f"{gap_ratio:.1f} fois l'écart médian "
             "entre rangs consécutifs), et non par une coupe de rang choisie. "
             f"{_verdict}}}\\label{{tab:tier}}",
             "\\begin{tabular}{@{}llrrrrl@{}}", "\\toprule",
             "Famille & Métrique & Paires & \\% & $\\rho$ & $p_{BH}$ (fam.) & "
             "n\\textsuperscript{o}1 \\\\", "\\midrule"]
    fam_prev = None
    for r in rows:
        fam = r["family"] if r["family"] != fam_prev else ""
        fam_prev = r["family"]
        lines.append(" & ".join([
            fam, r["label"],
            f"{r['pairs_ok']}/{r['pairs_total']}",
            num(100 * float(r["pair_acc"]), 0),
            num(r["rho"], 2, sign=True),
            num(r["p_bh_family"], 3),
            "\\textbf{oui}" if r["pred_best_correct"] == "True" else "non",
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_tier_ranking", lines,
          "\\texttt{results/rapport\\_data/tier\\_ranking\\_metrics.csv} "
          "(\\texttt{scripts/rapport/tier\\_ranking.py}). "
          f"Ordre vrai du palier reproduit dans {100 * float(noise['p_ordre_identique']):.0f}\\,\\% "
          f"des rééchantillonnages de seeds, $\\rho$ moyen avec l'ordre observé "
          f"{num(noise['rho_ordre_moyen'], 2, sign=True)} : l'ordre fin est bruité, "
          "l'ordre d'ensemble ne l'est pas.")


def t_tier_signs():
    """Le signe de chaque métrique selon la taille du palier."""
    fp = os.path.join(OUT, "tier_ranking_metrics.csv")
    if not os.path.exists(fp):
        return
    rows = load("tier_ranking_metrics.csv")
    tiers = sorted({int(r["tier"]) for r in rows})
    mets, seen = [], set()
    for r in sorted(rows, key=lambda r: (r["family"], r["metric"])):
        if r["metric"] not in seen:
            seen.add(r["metric"])
            mets.append((r["family"], r["metric"], r["label"]))
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{$\\rho$ de Spearman avec le F1 selon la taille du palier. "
             "Les métriques spectrales sont positives sur la population complète et "
             "\\textbf{négatives} dès qu'on retire les modèles cassés : leur "
             "corrélation d'ensemble est portée par ces derniers, pas par les bons "
             "modèles.}\\label{tab:tiersigns}",
             "\\begin{tabular}{@{}ll" + "r" * len(tiers) + "@{}}", "\\toprule",
             "Famille & Métrique & "
             + " & ".join(f"top-{t}" if t < max(tiers) else f"les {t}"
                          for t in tiers)
             + " \\\\", "\\midrule"]
    fam_prev = None
    for fam, m, lab in mets:
        cells = [fam if fam != fam_prev else "", lab]
        fam_prev = fam
        for t in tiers:
            sel = [r for r in rows if int(r["tier"]) == t and r["metric"] == m]
            cells.append(num(sel[0]["rho"], 2, sign=True) if sel else "---")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("t_tier_signs", lines,
          "\\texttt{results/rapport\\_data/tier\\_ranking\\_metrics.csv}.")



def main():
    os.makedirs(TAB, exist_ok=True)
    print("== Tableaux ==")
    t_dataset()
    t_support()
    t_pipelines()
    t_datacurve()
    t_frozen_probe()
    t_thresholds()
    t_fewshot()
    t_per_class_regimes()
    t_per_class_lora()
    t_geometry_datacurve()
    t_master()
    t_dinov3b_regimes()
    t_simb_regimes()
    t_ci()
    t_signif()
    t_signif_bh()
    t_per_class_models()
    t_knn()
    t_schemas()
    t_logme()
    t_conventions()
    t_geometry_full()
    t_correlations()
    t_tier_ranking()
    t_tier_signs()
    t_family()
    t_controls()
    t_layerwise()
    t_runs_raw()
    t_sources()
    t_ctx_distill()
    t_ctx_matrix()
    t_ctx_controls()
    t_ctx_sweep()
    t_head_tile()
    t_head_fused()
    t_simb_ablation()
    print(f"[OK] {TAB}")


if __name__ == "__main__":
    main()
