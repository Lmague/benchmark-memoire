#!/usr/bin/env python3
"""Figures « performances » du dossier rapport_bouguessa.

  perf_ranking_ci.png       classement des modèles, F1 + σ inter-seed + IC95
  perf_per_class_heatmap.png  F1 par classe × modèle
  perf_support_vs_f1.png    F1 d'une classe vs son support test
  perf_knn_vs_probe.png     probe linéaire vs k-NN
  perf_confusion.png        matrices de confusion, 6 modèles clés
  perf_schemas.png          F1 selon le schéma de classes (12 / 11 / 8)

    python3 rapport_bouguessa/figs/make_perf_figs.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "rapport"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import vizstyle as V
from registry import TIER_DISPLAY_SHORT
from registry import CLASSES_11, CLASSES_DEAD, DEPRECATED, EXCLUDED_MODELS

DATA = os.path.join(_ROOT, "results", "rapport_data")

# Couleur par TYPE de modèle (4 catégories, ordre fixe) — l'entité est le type
TYPE_COLOR = {"frozen": V.SERIES[0], "ft_fresh": V.SERIES[1],
              "ft_old": V.SERIES[2], "scratch": V.SERIES[4]}
TYPE_LABEL = {"frozen": "Gelé (probing linéaire)",
              "ft_fresh": "Affiné — init. SSL (DINOv3-B / SimDINOv2-B)",
              "ft_old": "Affiné — init. ImageNet",
              "scratch": "Depuis zéro"}


def load(fn, root=DATA):
    with open(os.path.join(root, fn)) as f:
        return list(csv.DictReader(f))


def keep(rows, field="model"):
    """Retire les modèles exclus (registry.EXCLUDED_MODELS) et dépréciés
    (registry.DEPRECATED, seed unique, « à ne jamais citer »)."""
    return [r for r in rows if r[field] not in EXCLUDED_MODELS
            and r[field] not in DEPRECATED]


def canonical_models():
    """F1 canonique + σ + type + dim (all_models_canonical_merged.json), modèles
    exclus des rapports retirés (registry.EXCLUDED_MODELS)."""
    d = json.load(open(os.path.join(_ROOT, "results",
                                    "all_models_canonical_merged.json")))["models"]
    out = []
    for m in d:
        if m["model"] in EXCLUDED_MODELS:
            continue
        t = m.get("type", "frozen")
        if "scratch" in m["model"]:
            t = "scratch"
        out.append({"model": m["model"], "display": m["display"], "type": t,
                    "f1": m["f1_linear_probe"], "std": m.get("f1_std"),
                    "seeds": m.get("f1_seeds"), "dim": m.get("dim"),
                    "n_seeds": m.get("n_seeds") or 1})
    out.sort(key=lambda r: r["f1"])
    return out


def bootstrap_ci():
    """IC95 par modèle, quand disponible (significance_matrix_all12.json)."""
    ci = {}
    fp = os.path.join(_ROOT, "results", "significance_matrix_all12.json")
    if os.path.exists(fp):
        d = json.load(open(fp))
        for k, v in d.get("model_stats", {}).items():
            if isinstance(v, dict) and "ci95_low" in v:
                ci[k] = (v["ci95_low"], v["ci95_high"])
    fp = os.path.join(_ROOT, "results", "significance_matrix_8group_fresh.json")
    if os.path.exists(fp):
        for g in json.load(open(fp)).get("groups", []):
            s = g.get("stats", {})
            if "ci95_low" in s:
                ci[g["name"]] = (s["ci95_low"], s["ci95_high"])
    return ci


# ── 1. classement ────────────────────────────────────────────────────────────
def fig_ranking():
    rows = canonical_models()
    ci = bootstrap_ci()
    # Forme : nuage de points (pas de barres) — l'écart utile tient dans une bande
    # de 0,13 et un axe de barres tronqué mentirait sur les rapports de longueur.
    fig, ax = plt.subplots(figsize=(9.2, max(7.2, 0.30 * len(rows) + 1.2)))
    xmin, xtext = 0.345, 0.516
    for i, r in enumerate(rows):
        c = TYPE_COLOR[r["type"]]
        if r["std"]:
            ax.errorbar(r["f1"], i, xerr=r["std"], color=c, lw=1.6, capsize=3,
                        zorder=3)
        if r["seeds"]:
            ax.plot(r["seeds"], [i] * len(r["seeds"]), ls="none", marker="|",
                    ms=8, color=c, alpha=0.75, zorder=4)
        ax.plot(r["f1"], i, marker="o", ms=7, color=c, zorder=5)
        lab = f"{r['f1']:.4f}".replace(".", ",")
        lab += f" ± {r['std']:.4f}".replace(".", ",") if r["std"] else "   (1 seed)"
        ax.text(xtext, i, lab, va="center", ha="left", fontsize=7.4, color=V.INK_SOFT)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{r['display']}  ({r['dim']}d)" for r in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlabel("F1-macro sur classes présentes — probe canonique, 11 classes")
    ax.set_xlim(xmin, 0.555)
    ax.set_xticks([0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525])
    ax.set_xticklabels(["0,350", "0,375", "0,400", "0,425", "0,450", "0,475", "0,500", "0,525"])
    V.clean(ax, y_only=False)
    ax.grid(axis="y", color=V.GRID, lw=0.5, ls=":")
    ax.grid(axis="x", visible=True)
    handles = [Line2D([], [], ls="none", marker="o", color=TYPE_COLOR[t],
                      label=TYPE_LABEL[t])
               for t in ("frozen", "ft_fresh", "ft_old", "scratch")]
    handles.append(Line2D([], [], color=V.INK_SOFT, marker="|", ls="-",
                          label="± σ inter-seed · traits = seeds individuels"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.8, bbox_to_anchor=(0.55, -0.045))
    ax.set_title(f"Classement des {len(rows)} modèles — probe linéaire canonique",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    V.savefig(fig, os.path.join(_HERE, "perf_ranking_ci.png"))
    _ = ci


# ── 2. F1 par classe × modèle ────────────────────────────────────────────────
def fig_per_class_heatmap():
    # Palier compétitif (probe interne des runs) : MÊME convention que la matrice
    # de significativité ; inclut les modèles de contexte et Vid-T-S/16.
    rows = load("per_class_tier.csv")
    M = np.array([[float(r[c]) for c in CLASSES_11] for r in rows])
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    im = ax.imshow(M, cmap=V.SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CLASSES_11)))
    ax.set_xticklabels([c + (" †" if c in CLASSES_DEAD else "") for c in CLASSES_11],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["display"] for r in rows], fontsize=7.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] > 0.005 else "·",
                    ha="center", va="center", fontsize=6.2,
                    color="white" if M[i, j] > 0.55 else V.INK)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cb.set_label("F1 de la classe", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"F1 par classe — palier compétitif ({len(rows)} modèles), probe interne des runs\n"
                 "† classes sans support test suffisant (ARCA 113, DRYI 177, RUBC 31 tuiles)",
                 loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    V.savefig(fig, os.path.join(_HERE, "perf_per_class_heatmap.png"))


# ── 3. support vs F1 ─────────────────────────────────────────────────────────
def fig_support_vs_f1():
    tiles = json.load(open(os.path.join(_ROOT, "results",
                                        "tiles_per_class_per_split.json")))
    rows = load("per_class_tier.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for c in CLASSES_11:
        x = tiles[c]["test"]
        vals = np.array([float(r[c]) for r in rows])
        col = V.SERIES[7] if c in CLASSES_DEAD else V.SERIES[0]
        ax.scatter([x] * len(vals), vals, s=14, color=col, alpha=0.45,
                   edgecolors="none", zorder=3)
        ax.scatter([x], [vals.mean()], s=70, color=col, marker="_", lw=2.2, zorder=4)
        ax.annotate(c, xy=(x, vals.max()), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=7.5, color=V.INK)
    ax.set_xscale("log")
    ax.set_xlabel("tuiles de test de la classe (échelle log)")
    ax.set_ylabel("F1 de la classe (un point = un modèle)")
    V.clean(ax)
    handles = [Line2D([], [], ls="none", marker="o", color=V.SERIES[0],
                      label="classe évaluable"),
               Line2D([], [], ls="none", marker="o", color=V.SERIES[7],
                      label="classe non évaluable (F1 ≡ 0)")]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title("Le support de test, pas le modèle, décide quelles classes sont "
                 "apprenables", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    V.savefig(fig, os.path.join(_HERE, "perf_support_vs_f1.png"))


# ── 4. k-NN vs probe ─────────────────────────────────────────────────────────
def fig_knn_vs_probe():
    rows = keep(load("knn_vs_probe.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    x = np.array([float(r["knn_best_f1"]) for r in rows])
    y = np.array([float(r["probe_f1_pres"]) for r in rows])
    lim = (min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02)
    ax.plot(lim, lim, color=V.RULE, ls="--", lw=1.1, zorder=1)
    ax.scatter(x, y, s=42, color=V.SERIES[0], zorder=3, edgecolors="none")
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(r["model"].replace("_", " "), xy=(xi, yi), xytext=(4, 3),
                    textcoords="offset points", fontsize=6.6, color=V.INK_SOFT,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                              alpha=0.75))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("F1 k-NN (meilleur k)")
    ax.set_ylabel("F1 probe linéaire")
    ax.set_title("Le probe linéaire domine partout le k-NN", loc="left",
                 fontweight="bold")
    V.clean(ax, y_only=False)

    ax = axes[1]
    ks = sorted(int(c.split("_")[1][1:]) for c in rows[0]
                if c.startswith("knn_k") and c.endswith("_f1") and rows[0][c] != "")
    for i, r in enumerate(sorted(rows, key=lambda r: -float(r["probe_f1_pres"]))[:6]):
        y = [float(r[f"knn_k{k}_f1"]) for k in ks]
        c = V.SERIES[i % len(V.SERIES)]
        ax.plot(ks, y, color=c, marker="o", ms=4, label=r["model"].replace("_", " "))
        ax.axhline(float(r["probe_f1_pres"]), color=c, ls=":", lw=1.0, alpha=0.55)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("F1 k-NN (trait plein) — probe en pointillé")
    ax.set_title("Sensibilité à k — 6 meilleurs modèles", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    V.clean(ax)
    fig.suptitle("Probe linéaire vs k plus proches voisins (test, 11 classes)",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    V.savefig(fig, os.path.join(_HERE, "perf_knn_vs_probe.png"))


# ── 5. matrices de confusion ─────────────────────────────────────────────────
def fig_confusion():
    # Palier compétitif (20 groupes) — même provenance que la significativité.
    fp = os.path.join(DATA, "confusion_tier.npz")
    if not os.path.exists(fp):
        print("  [SKIP] confusion_tier.npz absent")
        return
    z = np.load(fp)
    meta = json.load(open(os.path.join(DATA, "confusion_tier_meta.json")))
    keys = sorted(meta, key=lambda k: -meta[k]["f1_macro_pres"])
    ncol = 5
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow))
    for ax, k in zip(axes.ravel(), keys):
        M = z[k]
        im = ax.imshow(M, cmap=V.SEQ_BLUE, vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES_11)))
        ax.set_yticks(range(len(CLASSES_11)))
        ax.set_xticklabels(CLASSES_11, rotation=90, fontsize=4.6)
        ax.set_yticklabels(CLASSES_11, fontsize=4.6)
        ax.set_title(f"{meta[k]['label']}  ·  F1 = "
                     f"{meta[k]['f1_macro_pres']:.4f}".replace(".", ","),
                     loc="left", fontsize=7, fontweight="bold")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.06:
                    ax.text(j, i, f"{M[i, j]:.2f}"[1:], ha="center", va="center",
                            fontsize=5, color="white" if M[i, j] > 0.55 else V.INK)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Matrices de confusion à 100 % des données — lignes = vraie classe, "
                 "normalisées par ligne", x=0.02, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    V.savefig(fig, os.path.join(_HERE, "perf_confusion.png"))


# ── 6. schémas de classes ────────────────────────────────────────────────────
def fig_schemas():
    fp = os.path.join(_ROOT, "results", "f1_learnable_12models.csv")
    rows = keep(load(os.path.basename(fp), os.path.dirname(fp)))
    rows.sort(key=lambda r: -float(r["f1_macro_11"]))
    labels = [r["model"].replace("_", " ") for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    mean_d = float(np.mean([float(r["f1_macro_learnable"]) - float(r["f1_macro_12"])
                            for r in rows]))
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    for i, (col, lab) in enumerate([("f1_macro_12", "12 classes (avec RHOL)"),
                                    ("f1_macro_11", "11 classes (sans RHOL)"),
                                    ("f1_macro_learnable", "8 classes évaluables")]):
        ax.bar(x + (i - 1) * w, [float(r[col]) for r in rows], width=w,
               color=V.SERIES[i], label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.6)
    ax.set_ylabel("F1-macro")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    V.clean(ax)
    ax.set_title(f"Le schéma de classes déplace le F1 de {mean_d:+.2f} en moyenne "
                 "sans changer le classement des modèles", loc="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    V.savefig(fig, os.path.join(_HERE, "perf_schemas.png"))


# ── 7. matrice de significativité (palier complet) ───────────────────────────
def _signif_matrix_axes(ax, subnames, idx_global, delta, mark, obs, lim, fs, cb):
    """Dessine une sous-matrice de significativité sur `ax`.

    `idx_global` : indices (dans le tri global par F1 décroissant) des modèles de la
    sous-matrice. `delta`/`mark` : matrices globales (Δ et rejet BH).
    `lim` : borne de couleur commune aux deux panneaux (comparables)."""
    m = len(subnames)
    dsub = delta[np.ix_(idx_global, idx_global)] * 100.0   # points de pourcentage
    im = ax.imshow(dsub, cmap=V.DIV_BR, vmin=-lim, vmax=lim)
    rejset = {(i, j) for (i, j), r in mark.items() if r}
    # Une cellule = couleur (Δ, signe et ordre de grandeur) + ★ si la paire est
    # rejetée par Benjamini-Hochberg. On n'affiche PAS le Δ dans chaque case : à
    # 17×17 (et plus) des nombres de 5 caractères se touchent et redeviennent
    # illisibles. Les Δ exacts sont dans le tableau des rejets BH, cette figure ne
    # fait que montrer la structure (qui diffère de qui).
    for a in range(m):
        for b in range(m):
            if a == b:
                ax.text(b, a, "·", ha="center", va="center", color=V.INK_SOFT,
                        fontsize=fs)
                continue
            rej = (idx_global[a], idx_global[b]) in rejset
            if rej:
                ax.text(b, a, "★", ha="center", va="center", fontsize=fs + 4,
                        fontweight="bold", color="white")
    # Étiquettes lisibles (révision 2026-09-08, retour lecteur) :
    # « rang · nom court (F1) ». Le rang est celui du classement du palier —
    # le même que la colonne « Rang » du tableau des IC95 (t_ci), où chaque
    # nom court est déployé en entier (backbone --- régime, état).
    lab = [f"{r+1} · {TIER_DISPLAY_SHORT.get(m, m)}  "
           f"({obs[m]*100:.2f})".replace(".", ",")
           for m, r in zip(subnames, idx_global)]
    ax.set_xticks(range(m)); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=11)
    ax.set_yticks(range(m)); ax.set_yticklabels(lab, fontsize=11)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if cb:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("$\\Delta$ F1 \u00d710$^{-2}$ (ligne $-$ colonne)", fontsize=8)
        cbar.outline.set_visible(False)


def fig_signif_matrix():
    """Matrice de significativité, remplacée par DEUX matrices lisibles.

    La population 33×33 avec des chiffres en 6~pt était illisible (révision
    2026-09-08). On coupe le classement en deux blocs — tête (rangs 1–17) et queue
    (rangs 18–33) — et on rapporte la comparaison haut-vs-bas à la prose : elle est
    favorable au haut par construction (Δ > 0) et détaillée dans le tableau des
    rejets par Benjamini-Hochberg."""
    fp = os.path.join(_ROOT, "results", "significance_matrix_tier.json")
    if not os.path.exists(fp):
        print("  [SKIP] significance_matrix_tier.json absent "
              "(lancer scripts/rapport/significance_tier.py)")
        return
    d = json.load(open(fp))
    names = [g["name"] for g in sorted(d["groups"],
                                       key=lambda g: -g["stats"]["observed"])]
    obs = {g["name"]: g["stats"]["observed"] for g in d["groups"]}
    n = len(names)
    delta = np.full((n, n), np.nan)
    mark = {}
    for v in d["pairs"].values():
        a, b = v["model_a"], v["model_b"]
        i, j = names.index(a), names.index(b)
        dd = v["delta_observed_a_minus_b"]
        delta[i, j], delta[j, i] = dd, -dd
        mark[(i, j)] = mark[(j, i)] = v["bh_reject"]
    lim = np.nanmax(np.abs(delta)) * 100.0

    half = (n + 1) // 2
    blocks = [
        ("perf_signif_matrix_head.png",
         f"Matrice de significativité (a) — tête du palier ({half} modèles, "
         f"{half * (half - 1) // 2} paires internes)",
         "La couleur porte le Δ (rouge = la ligne bat la colonne) ; ★ = paire "
         f"rejetée par Benjamini-Hochberg ({d['bh_n_rejected']} sur {len(d['pairs'])} "
         "paires, les deux blocs). Rangs = tableau des IC95 ; Δ exacts = "
         "tableaux « rejets BH ».",
         names[:half], (11.4, 10.1)),
        ("perf_signif_matrix_tail.png",
         f"Matrice de significativité (b) — queue du palier ({n - half} modèles, "
         f"{(n - half) * (n - half - 1) // 2} paires internes)",
         "Le gros bloc sans ★ au centre est le palier plat SimDINOv2-B : les bras "
         "d'adaptation ne se distinguent pas.",
         names[half:], (11.0, 9.9)),
    ]
    for fn, title, note, subnames, figsize in blocks:
        fig, ax = plt.subplots(figsize=figsize)
        idx_global = [names.index(m) for m in subnames]
        _signif_matrix_axes(ax, subnames, idx_global, delta, mark, obs, lim,
                            fs=10 if len(subnames) <= 18 else 8, cb=True)
        # Pas de titre dans le PNG : la légende LaTeX le porte déjà, et un
        # titre long force la figure en paysage (labels alors minuscules à
        # l'échelle \textwidth). 2026-09-08.
        # (title/note conservés dans les tuples pour référence/documentation)
        _ = title, note
        fig.tight_layout()
        V.savefig(fig, os.path.join(_HERE, fn))
        plt.close(fig)


def main():
    V.setup()
    fig_ranking()
    fig_per_class_heatmap()
    fig_support_vs_f1()
    fig_knn_vs_probe()
    fig_confusion()
    fig_schemas()
    fig_signif_matrix()


if __name__ == "__main__":
    main()
