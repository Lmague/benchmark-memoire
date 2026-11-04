#!/usr/bin/env python3
"""Génère les deux tables LaTeX de l'ablation de rang LoRA et du coût/perf.

  t_lora_rank.tex   ablation r=2..32 (F1 canonique ± std, Δ gelé, best_epoch)
  t_cost_perf.tex   coût (params, FLOPs, F1, ratios) B-gelé/Full/MHSA/LoRA/H+gelé

Sources :
  - results/lora_rank_ablation_CANONICAL.json (probe canonique, scripts/rapport/
    lora_rank_ablation_probe.py) pour l'ablation ;
  - registry.CANONICAL_F1 et scripts/find_best_model.py (COST) pour les coûts ;
  - FLOPs de DINOv3 ViT-H+/16 = ESTIMATION par scaling linéaire (cf. note).

Format : booktabs, nombres français (virgule), ± pour l'écart-type, \textbf{}
pour le meilleur, caption en français, label unique, note « Source : ».

    python3 scripts/rapport/make_lora_rank_tables.py
"""
from __future__ import annotations

import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
sys.path.insert(0, _THIS)

from registry import CANONICAL_F1  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "rapport_bouguessa", "tables")
JSON = os.path.join(_ROOT, "results", "lora_rank_ablation_CANONICAL.json")

DIM = 768
N_BLOCKS = 12
N_PROJ = 2  # q et v (src/config.py target_modules=["q","v"])


def fr(x, nd=4, sign=False):
    """Nombre français : format .ndf puis virgule décimale (signe + optionnel)."""
    s = f"{x:.{nd}f}"
    if sign and x >= 0:
        s = "+" + s
    return s.replace(".", ",")


def lora_trainable(r):
    """Params LoRA entraînables = 2 (A+B) × dim × r × 2 (q,v) × 12 blocs."""
    return 2 * DIM * r * N_PROJ * N_BLOCKS


def t_lora_rank(data):
    ranks = data["ranks"]
    frozen = data["baselines"]["dinov3_vitb16_lvd"]["f1"]
    rows = []
    best_r = None
    best_f1 = -1.0
    for r in sorted((int(k) for k in ranks), key=int):
        d = ranks[str(r)]
        if d["f1_mean"] > best_f1:
            best_f1, best_r = d["f1_mean"], r
    for r in sorted((int(k) for k in ranks), key=int):
        d = ranks[str(r)]
        p = lora_trainable(r) / 1e6
        delta = d["f1_mean"] - frozen
        bold = "\\textbf{" if r == best_r else ""
        bend = "}" if r == best_r else ""
        rows.append(
            f"  {bold}{r}{bend} & {d['alpha']} & {fr(p, 3)} & "
            f"{bold}{fr(d['f1_mean'], 4)} $\\pm$ {fr(d['f1_std'], 4)}{bend} & "
            f"{fr(delta, 4, sign=True)} & {fr(d['best_epoch_mean'], 1)} \\\\"
        )
    r8 = ranks["8"]
    gap_note = (f"Le re-probe canonique de r=8 ici ({fr(r8['f1_mean'], 4)}) diffère "
                f"légèrement du F1 publié 0,4835 (provenance des runs différente).")
    tex = f"""\\begin{{table}}[H]
\\centering
\\small
\\caption{{Ablation du rang LoRA de DINOv3 ViT-B/16 LVD ($\\alpha = 2r$, 3 seeds). F1 = probe linéaire canonique mono-thread sur le test (11 classes sans RHOL), C sélectionné sur validation. $\\Delta$ = écart au backbone gelé (0,4712). Params LoRA entraînables = $2 \\times 768 \\times r \\times 2\\ (q,v) \\times 12\\ \\mathrm{{blocs}}$.}}\\label{{tab:lora_rank}}
\\begin{{tabular}}{{@{{}}rrrrrr@{{}}}}
\\toprule
Rang $r$ & $\\alpha$ & Params LoRA (M) & F1 moyen $\\pm$ std & $\\Delta$ gelé & Best epoch moyen \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\par\\vspace{{2pt}}\\footnotesize Source : re-probe canonique des embeddings \\texttt{{LoRA\\_ablation}} (\\texttt{{scripts/rapport/lora\\_rank\\_ablation\\_probe.py}}, \\texttt{{results/lora\\_rank\\_ablation\\_CANONICAL.json}}). {gap_note} best\\_epoch issu du \\texttt{{metrics.json}} du run (probe interne, non re-probé).
\\end{{table}}
"""
    return tex


def vit_gflops(depth, dim, n_heads, ffn_ratio, swiglu, n_tokens=197):
    """FLOPs forward-pass d'un ViT (attention + FFN), formule standard.

    Calibrée sur DINOv3-B et DINOv3-L (mêmes hypothèses : MLP classique,
    ffn_ratio=4) : reproduit les 17,6 / 61,6 GFLOPs du registre COST
    (scripts/find_best_model.py) à moins de 1,2 % près, sans facteur de
    correction. Utilisée telle quelle pour DINOv3 ViT-H+/16, dont les
    specs (depth=32, dim=1280, heads=20, ffn_ratio=6, SwiGLU) viennent du
    code officiel : facebookresearch/dinov3, dinov3/hub/backbones.py,
    fonction dinov3_vith16plus().
    """
    attn = n_tokens * dim * dim * 4 + 2 * n_tokens * n_tokens * dim
    ffn_mult = 3 if swiglu else 2  # SwiGLU = 3 projections (gate/up/down)
    ffn = ffn_mult * n_tokens * dim * (ffn_ratio * dim)
    return depth * (attn + ffn) / 1e9


def t_cost_perf(data):
    frozen = CANONICAL_F1["dinov3_vitb16_lvd"][0]
    full = CANONICAL_F1["dinov3_vitb16_lvd_full"][0]
    mhsa = CANONICAL_F1["dinov3_vitb16_lvd_mhsa"][0]
    lora8 = CANONICAL_F1["dinov3_vitb16_lvd_lora_r8"][0]
    hplus = CANONICAL_F1["dinov3_vith16plus_lvd"][0]

    # Vérification (non affichée) : la formule reproduit bien B et L.
    check_b = vit_gflops(12, 768, 12, 4, swiglu=False)   # -> 17,4 (registre : 17,6)
    check_l = vit_gflops(24, 1024, 16, 4, swiglu=False)  # -> 61,4 (registre : 61,6)
    assert abs(check_b - 17.6) / 17.6 < 0.02
    assert abs(check_l - 61.6) / 61.6 < 0.02

    hplus_gflops = vit_gflops(32, 1280, 20, 6.0, swiglu=True)  # -> 230,4

    lora_train = lora_trainable(8) / 1e6  # ≈ 0,295 M
    lora_total = 86.0 + lora_train        # ≈ 86,3 M (total, backbone gelé inclus)

    models = [
        ("DINOv3-B gelé", 86.0, 17.6, frozen),
        ("DINOv3-B Full FT", 86.0, 17.6, full),
        ("DINOv3-B MHSA", 86.0, 17.6, mhsa),
        ("DINOv3-B LoRA r=8", lora_total, 17.6, lora8),
        ("DINOv3 ViT-H+/16 gelé", 840.0, hplus_gflops, hplus),
    ]
    ratios = [(m[3] / m[1] * 100, m[3] / m[2] * 100) for m in models]
    best_f1 = max(m[3] for m in models)
    best_pp = max(r[0] for r in ratios)
    best_pg = max(r[1] for r in ratios)

    rows = []
    for i, (name, params, gflops, f1) in enumerate(models):
        pp, pg = ratios[i]
        bf = "\\textbf{" if f1 == best_f1 else ""
        be = "}" if f1 == best_f1 else ""
        bp = "\\textbf{" if pp == best_pp else ""
        bep = "}" if pp == best_pp else ""
        bg = "\\textbf{" if pg == best_pg else ""
        beg = "}" if pg == best_pg else ""
        rows.append(
            f"  {name} & {fr(params, 1)} & {fr(gflops, 1)} & {bf}{fr(f1, 4)}{be} & "
            f"{bp}{fr(pp, 3)}{bep} & {bg}{fr(pg, 3)}{beg} \\\\"
        )

    tex = f"""\\begin{{table}}[H]
\\centering
\\small
\\caption{{Coût et efficacité des régimes de DINOv3 ViT-B/16 contre le backbone DINOv3 ViT-H+/16 gelé. F1 = probe canonique (11 classes sans RHOL). FLOPs = coût d'inférence, fonction de l'architecture seule (identique pour Full/MHSA/LoRA sur un même backbone) ; LoRA ajoute {fr(lora_train, 3)}~M de paramètres entraînables mais aucune couche.}}\\label{{tab:cost_perf}}
\\begin{{tabular}}{{@{{}}l r r r r r@{{}}}}
\\toprule
Modèle & Params (M) & FLOPs (G) & F1 & F1/params ($\\times$100) & F1/GFLOP ($\\times$100) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\par\\vspace{{2pt}}\\footnotesize FLOPs calculés depuis l'architecture officielle (depth=32, dim=1280, ffn\\_ratio=6, SwiGLU~; \\texttt{{facebookresearch/dinov3}}, \\texttt{{hub/backbones.py}}), formule standard ViT (attention + FFN, 197 tokens) ; reproduit les GFLOPs de ViT-B/16 et ViT-L/16 du registre à moins de 1,2\\,\\% près (mêmes hypothèses, ffn\\_ratio=4, MLP classique).
\\end{{table}}
"""
    return tex


def main():
    data = json.load(open(JSON))
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn, content in (("t_lora_rank.tex", t_lora_rank(data)),
                        ("t_cost_perf.tex", t_cost_perf(data))):
        out = os.path.join(OUT_DIR, fn)
        with open(out, "w") as f:
            f.write(content)
        print(f"  [TAB] {out} ({len(content)} octets)")


if __name__ == "__main__":
    main()
