#!/usr/bin/env python3
"""Vérifie le run canonique : tableau des bras, critères G1-G4, comparaison aux
références de la clean baseline (R0 = 0,4717 tuile seule ; R2 = 0,5059 contexte 512).

Aucune valeur n'est écrite en dur pour les bras : tout vient des JSON produits.
Les deux références sont des constantes explicitement étiquetées.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

REF = {"R0": ("tile seule, SimDINOv2 ViT-B/16, probe canonique", 0.4717),
       "R2": ("contexte 512 px, fusion (reference clean baseline)", 0.5059)}


def load(out_dir: str) -> dict:
    arms = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "arms", "*.json"))):
        d = json.load(open(p))
        arms[d["arm"]] = d
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    arms = load(a.out_dir)
    m8_path = os.path.join(a.out_dir, "metrics_8cls.json")
    m8 = json.load(open(m8_path)) if os.path.exists(m8_path) else {}
    boot = json.load(open(os.path.join(a.out_dir, "bootstrap.json")))

    print(f"{'bras':5s} {'F1 11cls':>9s} {'±':>7s} {'F1 8cls':>8s}")
    for name in ("R0", "R1", "R2", "R3", "R4", "L1", "L2", "L1p"):
        if name not in arms:
            continue
        m = arms[name]["metrics"]
        sd = m.get("f1_macro_pres_std")
        f8 = m8.get(name, {}).get("f1_macro_8cls")
        print(f"{name:5s} {m['f1_macro_pres']:9.4f} {(sd if sd is not None else 0):7.4f} "
              f"{(f8 if f8 is not None else float('nan')):8.4f}")

    print("\n--- comparaison aux références canoniques ---")
    for k, (label, ref) in REF.items():
        if k in arms:
            got = arms[k]["metrics"]["f1_macro_pres"]
            print(f"{k}: {got:.4f} vs {ref:.4f} (réf. {label})  Δ={got - ref:+.4f}")

    print("\n--- critères pré-enregistrés ---")
    f = {k: v["metrics"]["f1_macro_pres"] for k, v in arms.items()}
    g1 = f.get("L1", float("nan")) - f.get("R2", float("nan"))
    print(f"G1  L1 - R2 > +0.010        : {g1:+.4f}  -> {'PASS' if g1 > 0.010 else 'ECHEC'}")
    g2 = f.get("L1", float("nan")) - f.get("L2", float("nan"))
    print(f"G2  L1 > L2 (apparié params) : {g2:+.4f}  -> {'PASS' if g2 > 0 else 'ECHEC'}")
    key = "L1-R2"
    pr = boot.get("paired", {})
    entry = pr.get("L1-R2")
    if entry:
        lo, hi = entry["ci95"]
        print(f"G3  IC L1-R2 exclut 0       : [{lo:+.4f}, {hi:+.4f}] p={entry['p_two_sided_boot']}  "
              f"-> {'PASS' if (lo > 0 or hi < 0) else 'ECHEC'}")
    if m8:
        g4 = m8.get("L1", {}).get("f1_macro_8cls", float("nan")) - m8.get("R3", {}).get("f1_macro_8cls", float("nan"))
        if g4 == g4:
            print(f"G4  L1 > R3 en 8 classes     : {g4:+.4f}  -> {'PASS' if g4 > 0 else 'ECHEC'}")


if __name__ == "__main__":
    main()
