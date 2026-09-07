#!/usr/bin/env python3
"""Remplit registry.CANONICAL_F1 avec les 13 bras Stage A (reprobe canonique) + les
2 runs SimDINOv2-B entraînés Design B (metrics.json, même provenance que les
entrées ctxdistill R1/R2/R3 du 2026-08-31), et normalise les libellés d'affichage
(plus d'Unicode dans les noms — les figs/tables les utilisent tels quels).

À lancer APRÈS scripts/rapport/probe_simb_stageA_canonical.py.

    python3 scripts/rapport/fill_registry_stageA.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys

_HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, _HERE)
from registry import p  # noqa: E402

ARMS = ["simdinov2_vitb16_lora_r8_b611", "simdinov2_vitb16_lora_r8_b05",
        "simdinov2_vitb16_lora_r8_b911", "simdinov2_vitb16_lora_r2",
        "simdinov2_vitb16_lora_r4", "simdinov2_vitb16_lora_r16",
        "simdinov2_vitb16_lora_r32", "simdinov2_vitb16_lora_r8_s2",
        "simdinov2_vitb16_lora_r16_s2", "simdinov2_vitb16_lora_r8_rslora",
        "simdinov2_vitb16_lora_r16_rslora", "simdinov2_vitb16_lora_r8_qkv",
        "simdinov2_vitb16_norm_tuning"]

DISPLAY_FIX = {
    "SimDINOv2-B LoRA r8 (rsLoRA, scaling √8)": "SimDINOv2-B LoRA r8 (rsLoRA)",
    "SimDINOv2-B LoRA r16 (rsLoRA, scaling √16)": "SimDINOv2-B LoRA r16 (rsLoRA)",
}

SIMB_TRAINED = {
    # clé registre : (display, tag_stem_runs, dim)
    "ctxdistill_dB_tSL_r2a4": (
        "Contexte SimB (B, fusion 512, r2a4)",
        "simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r2a4_frac100", 1536),
    "ctxdistill_dB_tSL_r8a16": (
        "Contexte SimB (B, fusion 512, r8a16)",
        "simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r8a16_frac100", 1536),
}


def main():
    canon = {m["model"]: m for m in json.load(
        open(p("results", "simb_stageA_probe_CANONICAL.json")))["models"]}
    missing = [a for a in ARMS if a not in canon]
    if missing:
        raise SystemExit(f"reprobe incomplet, bras absents : {missing}")

    entries = {}
    for a in ARMS:
        m = canon[a]
        entries[a] = (round(m["f1_linear_probe"], 6), round(m["f1_std"], 6))

    trained = {}
    for key, (display, stem, dim) in SIMB_TRAINED.items():
        seeds = [json.load(open(p("results", "context_distill", "runs",
                                           f"{stem}_seed{s}", "metrics.json")))
                 ["f1_macro_pres_test"] for s in (0, 1, 2)]
        trained[key] = {
            "model": key, "type": "ft_fresh", "display": display,
            "f1_linear_probe": float(statistics.mean(seeds)),
            "f1_std": float(statistics.stdev(seeds)),
            "f1_seeds": seeds,
            "silhouette_score": None, "silhouette_std": None,
            "nc1": None, "nc2_deviation_etf": None,
            "dim": dim, "n_train": 49281, "n_test": 17598,
            "n_classes": 11, "n_seeds": 3,
            "f1_macro_all": float(statistics.mean(seeds)),
        }
        entries[key] = (round(trained[key]["f1_linear_probe"], 6),
                        round(trained[key]["f1_std"], 6))

    # 1. fusion trained dans all_models_canonical_merged.json (+ normalisation √)
    mp = p("results", "all_models_canonical_merged.json")
    merged = json.load(open(mp))
    have = {m["model"] for m in merged["models"]}
    for m in merged["models"]:
        if m["display"] in DISPLAY_FIX:
            m["display"] = DISPLAY_FIX[m["display"]]
    for key, rec in trained.items():
        if key not in have:
            merged["models"].append(rec)
    merged["models"].sort(key=lambda m: -m["f1_linear_probe"])
    merged["n_models"] = len(merged["models"])
    import time
    merged["date"] = time.strftime("%Y-%m-%d")
    json.dump(merged, open(mp, "w"), indent=2)

    # 2. patch registry.py CANONICAL_F1 (remplace le marqueur prévu à cet effet)
    rp = p("scripts", "rapport", "registry.py")
    txt = open(rp).read()
    lines = ["    # ── ajout 2026-09 : Stage A SimDINOv2-B (13 bras, reprobe canonique",
             "    # results/simb_stageA_probe_CANONICAL.json) + 2 SimDINOv2-B entraînés",
             "    # Design B (metrics.json, même provenance que ctxdistill R1/R2/R3) ──"]
    for a in ARMS:
        f1, sd = entries[a]
        lines.append(f"    {a!r}: ({f1:.4f}, {sd:.4f}),")
    for key in SIMB_TRAINED:
        f1, sd = entries[key]
        lines.append(f"    {key!r}: ({f1:.4f}, {sd:.4f}),  # metrics.json, pas de bootstrap (NO_BOOTSTRAP)")
    block = "\n".join(lines) + "\n"
    marker = "    # __STAGEA_FILL__\n"
    assert marker in txt, "marqueur __STAGEA_FILL__ absent de registry.py"
    txt = txt.replace(marker, block)
    open(rp, "w").write(txt)

    print("registre rempli :")
    for k in list(ARMS) + list(SIMB_TRAINED):
        print(f"  {k:38s} {entries[k][0]:.4f} ± {entries[k][1]:.4f}")
    import subprocess
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0, 'scripts/rapport');"
                        "from registry import CANONICAL_F1, tier_break, TIER_K;"
                        f"print('TIER_K actuel =', TIER_K, '| tier_break() =', tier_break())"],
                       capture_output=True, text=True, cwd=p())
    print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
