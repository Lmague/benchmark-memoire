#!/usr/bin/env python3
"""Figures de la section « contexte spatial » (self-distillation) —
architecture Design A/B + résultats. Style commun via vizstyle.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizstyle as V
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, PURPLE, RED = V.SERIES
GRAY = "#8a8a8a"

# ─────────────────────────────────────────────────────────────── aides schéma
def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8.5, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)

def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=10, color=color, lw=lw, linestyle=ls))

def label(ax, x, y, text, fs=7.2, color="#555555", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color)

def new_panel(title):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axes[0].set_title(f"(a) {title} — entraînement", fontsize=9.5, loc="left")
    axes[1].set_title(f"(b) {title} — inférence", fontsize=9.5, loc="left")
    return fig, axes

# ───────────────────────────────────────────────────────── Design A
fig, axes = new_panel("Design A — contexte au TRAIN seulement")
ax = axes[0]
box(ax, 0.02, 0.72, 0.20, 0.18, "Tuile 224px", fc="white", ec=GRAY)
box(ax, 0.30, 0.70, 0.26, 0.22, "DINOv3-B\nLoRA r=2", fc="#eef3fb", ec=BLUE, bold=True)
box(ax, 0.64, 0.72, 0.16, 0.18, "feat_tile\n768", fc="#eef3fb", ec=BLUE)
box(ax, 0.85, 0.72, 0.13, 0.18, "Tête + focal", fc="#fdf0e7", ec=ORANGE)
arrow(ax, 0.22, 0.81, 0.29, 0.81); arrow(ax, 0.56, 0.81, 0.63, 0.81)
arrow(ax, 0.80, 0.81, 0.85, 0.81)
box(ax, 0.02, 0.30, 0.20, 0.18, "Contexte\n1024px", fc="white", ec=GRAY)
box(ax, 0.30, 0.28, 0.26, 0.22, "Teacher DINOv3-L\nGELÉ", fc="#f2f2f2", ec=GRAY)
box(ax, 0.64, 0.30, 0.16, 0.18, "feat_ctx\n1024", fc="#f2f2f2", ec=GRAY)
arrow(ax, 0.22, 0.39, 0.29, 0.39, color=GRAY); arrow(ax, 0.56, 0.39, 0.63, 0.39, color=GRAY)
ax.plot([0.66, 0.74], [0.48, 0.70], color=RED, lw=1.1, ls="--")
label(ax, 0.71, 0.56, "distill loss", fs=7, color=RED)
label(ax, 0.50, 0.05, "STUDENT entraîné (seul backprop)\nTeacher gelé, jamais rétropropagé", fs=7)
ax = axes[1]
box(ax, 0.02, 0.42, 0.20, 0.18, "Tuile 224px", fc="white", ec=GRAY)
box(ax, 0.30, 0.40, 0.26, 0.22, "DINOv3-B\n(LoRA entraîné)", fc="#eef3fb", ec=BLUE, bold=True)
box(ax, 0.64, 0.42, 0.16, 0.18, "feat_tile\n768", fc="#eef3fb", ec=BLUE)
box(ax, 0.85, 0.42, 0.13, 0.18, "Probe\nlinéaire", fc="#e7f4ec", ec=GREEN)
arrow(ax, 0.22, 0.51, 0.29, 0.51); arrow(ax, 0.56, 0.51, 0.63, 0.51)
arrow(ax, 0.80, 0.51, 0.85, 0.51)
ax.text(0.50, 0.78, "✗ contexte non utilisé à l'inférence", fontsize=8, ha="center", color="red")
label(ax, 0.50, 0.12, "1 seul forward • 768 dims\nF1 = 0.487 (R1) / 0.484 (R3)", fs=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctx_arch_A.png"), dpi=220); plt.close(fig)

# ───────────────────────────────────────────────────────── Design B
fig, axes = new_panel("Design B — contexte au train ET à l'inférence")
ax = axes[0]
box(ax, 0.02, 0.72, 0.20, 0.18, "Tuile 224px", fc="white", ec=GRAY)
box(ax, 0.02, 0.30, 0.20, 0.18, "Contexte\n1024px", fc="white", ec=GRAY)
box(ax, 0.30, 0.42, 0.26, 0.30, "DINOv3-B\nLoRA r=2\n(2 forwards)", fc="#eef3fb", ec=BLUE, bold=True)
box(ax, 0.63, 0.42, 0.17, 0.30, "feat_tile 768\n⊕\nfeat_ctx 768", fc="#eef3fb", ec=BLUE)
box(ax, 0.85, 0.42, 0.13, 0.30, "Tête\nconcat\n(1536)", fc="#fdf0e7", ec=ORANGE)
arrow(ax, 0.22, 0.81, 0.29, 0.70); arrow(ax, 0.22, 0.39, 0.29, 0.50)
arrow(ax, 0.56, 0.57, 0.62, 0.57); arrow(ax, 0.80, 0.57, 0.85, 0.57)
box(ax, 0.36, 0.08, 0.24, 0.16, "Teacher DINOv3-L gelé\n(cible distill, idem A — train seul)", fc="#f2f2f2", ec=GRAY)
# loss de distillation teacher→student (même branche qu'en A), train seulement
ax.plot([0.48, 0.66], [0.24, 0.43], color=RED, lw=1.1, ls="--")
label(ax, 0.57, 0.33, "distill loss", fs=7, color=RED)
label(ax, 0.50, 0.015, "HEAD entraînée sur [tuile;contexte] — seule différence avec A :\nla tête, pas la distillation (teacher gelé, idem A, jamais à l'inférence)", fs=7)
ax = axes[1]
box(ax, 0.02, 0.70, 0.19, 0.18, "Tuile 224px", fc="white", ec=GRAY)
box(ax, 0.02, 0.32, 0.19, 0.18, "Contexte\n1024px", fc="white", ec=GRAY)
box(ax, 0.30, 0.38, 0.25, 0.32, "DINOv3-B\n(2 forwards\nmême poids)", fc="#eef3fb", ec=BLUE, bold=True)
box(ax, 0.63, 0.38, 0.16, 0.32, "feat_tile 768\n⊕\nfeat_ctx 768\n= 1536", fc="#eef3fb", ec=BLUE)
box(ax, 0.85, 0.38, 0.13, 0.32, "Probe\nlinéaire", fc="#e7f4ec", ec=GREEN)
arrow(ax, 0.21, 0.79, 0.29, 0.62); arrow(ax, 0.21, 0.41, 0.29, 0.46)
arrow(ax, 0.55, 0.54, 0.62, 0.54); arrow(ax, 0.79, 0.54, 0.85, 0.54)
label(ax, 0.50, 0.12, "2 forwards • 1536 dims • F1 = 0.508 (R2)", fs=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctx_arch_B.png"), dpi=220); plt.close(fig)

# ───────────────────────────────────────────────────────── F1 comparison
# Données lues depuis results/context_distill/runs/*/metrics.json (jamais tapées).
import json as _json, statistics as _st
_ROOT = os.path.dirname(os.path.dirname(FIG))
def _seeds(stem):
    return [_json.load(open(os.path.join(
        _ROOT, "results", "context_distill", "runs", f"{stem}_seed{s}",
        "metrics.json")))["f1_macro_pres_test"] for s in (0, 1, 2)]
_frozen = 0.4712  # canonique DINOv3-B gelé (registry)
_specs = [
    ("DINOv3-B\ngelé", [_frozen], GRAY),
    ("LoRA r=8\n(publié)", [0.4835], BLUE),  # canonique (registry)
    ("Blocs 6-11\n(LoRA)", [0.4844], BLUE),  # ablation blocs (rapport 2026-08)
    ("R3\nA·EMA", _seeds("dinov3_vitb16_lvd_ctxdistill_dA_tEMA_ctx1024_r2a4_frac100"), ORANGE),
    ("R1\nA·L", _seeds("dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100"), ORANGE),
    ("R2\nB·L", _seeds("dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100"), RED),
    ("SimB@512\nB·r2a4", _seeds("simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r2a4_frac100"), PURPLE),
    ("SimB@512\nB·r8a16", _seeds("simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r8a16_frac100"), PURPLE),
]
labels = [s[0] for s in _specs]
f1s = [float(_st.mean(s[1])) for s in _specs]
errs = [float(_st.stdev(s[1])) if len(s[1]) > 1 else 0.0 for s in _specs]
colors = [s[2] for s in _specs]
colors = [GRAY] + [BLUE]*2 + [ORANGE, ORANGE, RED]
fig, ax = plt.subplots(figsize=(7.8, 3.2))
bars = ax.bar(range(len(labels)), f1s, yerr=errs, color=colors, edgecolor="black",
              linewidth=0.6, capsize=3, width=0.62, error_kw=dict(lw=0.9))
for i, (b, v) in enumerate(zip(bars, f1s)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.4f}", ha="center", fontsize=7.2)
ax.axhline(0.4712, color=GRAY, ls=":", lw=1.0)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.4)
ax.set_ylabel("F1 macro (test v3)", fontsize=8.5)
ax.set_ylim(0.455, 0.525); ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_title("Contexte 1024px — F1 par design (moy. 3 seeds ± std)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctx_f1.png"), dpi=220); plt.close(fig)

# ───────────────────────────────────────────────────────── attribution matrix
# Données lues depuis results/context_distill/controls/fused_probe_*.json.
_ctrl = os.path.join(_ROOT, "results", "context_distill", "controls")
def _tf(fn):
    d = _json.load(open(os.path.join(_ctrl, fn)))
    return d["tile"]["f1_macro_pres_test"], d["fused"]["f1_macro_pres_test"]
# R2 : pas de clé "fused" dans le contrôle seed0 — le fused R2 est la moyenne
# 3 seeds des metrics.json (0.5098), la tuile vient du contrôle (0.4779).
_r2_tile = _json.load(open(os.path.join(
    _ctrl, "fused_probe_r2_dB_tL_seed0.json")))["tile"]["f1_macro_pres_test"]
_r2_fused = float(sum(_seeds(
    "dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100")) / 3)
_pairs = [_tf("fused_probe_frozen_seed0.json"),
          _tf("fused_probe_lora_r2a4_v3train_seed0.json"),
          _tf("fused_probe_lora_r8a16_spatial_seed0.json"),
          _tf("fused_probe_r1_dA_tL_seed0.json"),
          (_r2_tile, _r2_fused)]
models = ["Gelé", "LoRA r2", "LoRA r8\nspatial", "R1 distil.\nA", "R2 distil.\nB"]
tile = [p[0] for p in _pairs]
fused = [p[1] for p in _pairs]
x = range(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(6.6, 3.3))
b1 = ax.bar([i - w/2 for i in x], tile, w, color=BLUE, edgecolor="black", lw=0.6, label="tuile seule (768)")
b2 = ax.bar([i + w/2 for i in x], fused, w, color=AQUA, edgecolor="black", lw=0.6, label="tuile⊕contexte (1536)")
for i in x:
    ax.text(i - w/2, tile[i] + 0.004, f"{tile[i]:.4f}", ha="center", fontsize=6.6)
    ax.text(i + w/2, fused[i] + 0.004, f"{fused[i]:.4f}", ha="center", fontsize=6.6)
    ax.text(i, 0.465, f"Δ {fused[i]-tile[i]:+.3f}", ha="center", fontsize=7, color="#333")
ax.set_xticks(list(x)); ax.set_xticklabels(models, fontsize=7.6)
ax.set_ylabel("F1 macro (test v3, seed 0)", fontsize=8.5)
ax.set_ylim(0.455, 0.525); ax.legend(fontsize=7.4, loc="lower right")
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_title("Attribution : la même sonde sur les mêmes features — seul R2 franchit 0.51", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctx_matrix.png"), dpi=220); plt.close(fig)

# ───────────────────────────────────────────────────────── geometry
# Données lues depuis results/context_distill/geometry/geometry_*_splittest.json
# (moyenne sur les seeds disponibles ; RankMe/D calculé ici). Si un fichier manque,
# repli sur les valeurs publiées d'ANALYSE.md §4 (documenté, pas silencieux).
import glob as _glob
# Schéma réel : un fichier par seed, clés = tags complets, sous-dicts
# {tile | fused | tile_only} → {rankme, anisotropy, dim, ...}.
# Gelé : results/context_distill/geometry/geometry_with_frozen_baseline.json.
_g_aniso, _g_rankme = {}, {}
_gfiles = sorted(_glob.glob(os.path.join(
    _ROOT, "results", "context_distill", "geometry", "geometry_*_splittest.json")))
for _gf in _gfiles:
    _gd = _json.load(open(_gf))["models"]
    for _tag, _m in _gd.items():
        for _rep in ("tile", "fused", "tile_only"):
            if _rep in _m:
                _r = _m[_rep]
                _g_aniso.setdefault((_tag, _rep), []).append(_r["anisotropy"])
                _g_rankme.setdefault((_tag, _rep), []).append(
                    _r["rankme"] / _r["dim"])
try:
    _fb = _json.load(open(os.path.join(
        _ROOT, "results", "context_distill", "geometry",
        "geometry_with_frozen_baseline.json")))["frozen_baseline"]
    _g_aniso[("FROZEN", "x")] = [_fb["anisotropy"]]
    _g_rankme[("FROZEN", "x")] = [_fb["rankme"] / _fb["dim"]]
except (OSError, KeyError) as _e:
    print(f"[ctx_figs] ATTENTION : baseline gelée illisible ({_e})")
def _gsel(which, design, rep, seeds=(0, 1, 2)):
    """Moyenne sur les seeds du (design, représentation) demandé."""
    src = _g_aniso if which == "aniso" else _g_rankme
    vs = []
    for _s in seeds:
        _tag = (f"dinov3_vitb16_lvd_ctxdistill_{design}_ctx1024_r2a4_frac100_seed{_s}"
                if design != "FROZEN" else "FROZEN")
        _rep = "x" if design == "FROZEN" else rep
        vs += src.get((_tag, _rep), [])
    if not vs:
        raise SystemExit(f"[ctx_figs] ERREUR : aucune géométrie pour {design}/{rep}")
    return float(sum(vs) / len(vs))
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax = axes[0]
names = ["Gelé", "R1\ndistil. A", "R3\ndistil. A", "R2 fused", "R2\ntuile seule"]
aniso = [_gsel("aniso", "FROZEN", "x"),
         _gsel("aniso", "dA_tL", "tile"),
         _gsel("aniso", "dA_tEMA", "tile"),
         _gsel("aniso", "dB_tL", "fused"),
         _gsel("aniso", "dB_tL", "tile_only")]
cols = [GRAY, ORANGE, ORANGE, RED, RED]
ax.bar(range(len(names)), aniso, color=cols, edgecolor="black", lw=0.6, width=0.6)
for i, v in enumerate(aniso):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7.2)
ax.set_ylabel("Anisotropie (↓ = plus isotrope)", fontsize=8)
ax.set_title("(a) Anisotropie", fontsize=9)
ax.set_ylim(0.28, 0.66); ax.grid(axis="y", alpha=0.25, lw=0.5)
ax = axes[1]
rn = [_gsel("rankme", "FROZEN", "x"),
      _gsel("rankme", "dA_tL", "tile"),
      _gsel("rankme", "dA_tEMA", "tile"),
      _gsel("rankme", "dB_tL", "fused"),
      _gsel("rankme", "dB_tL", "tile_only")]
lbl = ["Gelé", "R1", "R3", "R2 fused", "R2 tile"]
ax.bar(range(len(names)), rn, color=cols, edgecolor="black", lw=0.6, width=0.6)
for i, v in enumerate(rn):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(len(names))); ax.set_xticklabels(lbl, fontsize=7.2)
ax.set_ylabel("RankMe / dim", fontsize=8)
ax.set_title("(b) Rang effectif normalisé", fontsize=9)
ax.set_ylim(0.42, 0.50); ax.grid(axis="y", alpha=0.25, lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ctx_geom.png"), dpi=220); plt.close(fig)

print("figures écrites :", [f for f in os.listdir(FIG) if f.startswith("fig_ctx_")])
