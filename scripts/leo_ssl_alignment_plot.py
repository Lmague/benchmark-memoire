#!/usr/bin/env python3
"""Figure dose-réponse d'alignement — F1 frozen vs époques de SSL aérien (Léo ExPLoRA).

Lit ``results/leo_explora_ssl/probes/*.json`` (produits par
``scripts/leo_ssl_alignment_probe.py``) et trace les trois compteurs
(tile / ctx / fused @512) en fonction de la dose de SSL aérien, moyenne ± écart-type
sur les 3 seeds, avec le point époque 0 (SimDINOv2-B iNat) et les références du
chapitre contexte en pointillés.

Usage :
    python scripts/leo_ssl_alignment_plot.py
    python scripts/leo_ssl_alignment_plot.py --out results/leo_explora_ssl/figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent

REF_FUSED = 0.5059   # SimB-iNat gelé-fusionné @512 (results/context_distill/CONTROLES_BOUGUESSA.md)
REF_TILE = 0.4717    # SimB-iNat gelé, tuile seule @512
REF_CTX = 0.4931     # SimB-iNat gelé, contexte seul @512
COLORS = {"tile": "#1f77b4", "ctx": "#2ca02c", "fused": "#d62728"}
LABELS = {"tile": "tuile seule (768)", "ctx": "contexte seul (768)",
          "fused": "fusion tuile⊕contexte @512 (1536)"}


def load_rows(probe_dir: Path) -> tuple[dict, dict]:
    """→ (séries {(variant, epoch): [f1, ...]}, baseline {variant: f1})."""
    series: dict = {}
    baseline: dict = {}
    for p in sorted(probe_dir.glob("leossl_b16_*_*.json")):
        d = json.loads(p.read_text())
        v, ep = d.get("variant"), d.get("epoch_0based")
        if d.get("seed") is None or ep is None:
            if v is not None:
                baseline[v] = d["f1_macro_pres_test"]
            continue
        series.setdefault((v, int(ep)), []).append(d["f1_macro_pres_test"])
    return series, baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default=str(PROJ / "results" / "leo_explora_ssl" / "probes"))
    ap.add_argument("--out", default=str(PROJ / "results" / "leo_explora_ssl" / "figures"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe_dir = Path(args.probes)
    series, baseline = load_rows(probe_dir)
    if not series:
        raise SystemExit(f"[plot] aucune sonde dans {probe_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for v in ("tile", "ctx", "fused"):
        eps = sorted(ep for (vv, ep) in series if vv == v)
        if not eps:
            continue
        # époque 0 = SimB iNat, puis époques finies : on n'ajoute le point 0 que si le
        # checkpoint de référence a bien été extrait (le job SLURM peut le sauter).
        xs, ys, es = [], [], []
        if v in baseline:
            xs.append(0); ys.append(baseline[v]); es.append(0.0)
        for ep in eps:
            a = np.asarray(series[(v, ep)], dtype=float)
            xs.append(ep + 1)
            ys.append(a.mean())
            es.append(a.std(ddof=1) if len(a) > 1 else 0.0)
        ys, es = np.asarray(ys), np.asarray(es)
        ax.plot(xs, ys, "-o", ms=4, color=COLORS[v], label=LABELS[v])
        ax.fill_between(xs, ys - es, ys + es, color=COLORS[v], alpha=0.18, linewidth=0)

    ax.axhline(REF_FUSED, color=COLORS["fused"], ls="--", lw=1.0,
               label=f"référence SimB-iNat gelé-fusionné @512 = {REF_FUSED:.4f}")
    ax.axhline(REF_TILE, color=COLORS["tile"], ls=":", lw=1.0,
               label=f"réf. SimB-iNat tuile seule = {REF_TILE:.4f}")
    ax.axhline(REF_CTX, color=COLORS["ctx"], ls=":", lw=1.0,
               label=f"réf. SimB-iNat contexte seul = {REF_CTX:.4f}")

    ax.set_xlabel("Époques de SSL aérien (Léo, ExPLoRA DINO+iBOT+KoLeo) — 0 = SimDINOv2-B iNat")
    ax.set_ylabel("F1-macro test (11 classes, split spatial v3)")
    ax.set_title("Dose-réponse d'alignement : SSL aérien Léo → sonde gelée sur Arctic-TVC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = out_dir / f"alignment_doseresponse.{ext}"
        fig.savefig(p, dpi=200)
        print(f"[plot] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
