#!/usr/bin/env python3
"""Construit la manifeste des inputs du head-sweep pour Narval.

Liste EXACTEMENT les fichiers .npy que `tile_head_sweep.py` lit via
`significance_tier.GROUPS` (chemins relatifs au dépôt), plus les résultats
déjà produits localement (pour skip-if-done de l'autre côté). Écrit :
  - /tmp/head_sweep_manifest.txt   (paths relatifs, un par ligne — pour rsync --files-from)
  - /tmp/head_sweep_size.txt       (taille totale lisible par humans)

Usage :  python3 scripts/prepare_head_sweep_push.py
Puis :   bash scripts/push_head_sweep_inputs.sh     (rsync, MFA requis)
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_HERE, "rapport"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402  (post-import usage only)


def needed_files():
    from significance_tier import GROUPS
    files = set()
    missing = []
    for name, kind, path, seeds, _ in GROUPS:
        if kind == "frozen":
            for s in ("train", "val", "test"):
                for f in (f"embeddings/{path}_{s}.npy",
                          f"embeddings/{path}_{s}_labels.npy"):
                    (files.add(f) if os.path.exists(os.path.join(ROOT, f))
                     else missing.append((name, f)))
        else:
            for sd in seeds:
                base = path.format(s=sd)
                for s in ("train", "val", "test"):
                    for f in (f"{base}/{s}.npy", f"{base}/{s}_labels.npy"):
                        (files.add(f) if os.path.exists(os.path.join(ROOT, f))
                         else missing.append((name, f)))
    return sorted(files), missing


def main():
    files, missing = needed_files()
    results = [os.path.relpath(os.path.join(dp, f), ROOT)
               for dp, _, fs in os.walk(os.path.join(ROOT, "results/rapport_data/tile_heads"))
               for f in fs if f.endswith(".json")]
    total = files + sorted(results)
    with open("/tmp/head_sweep_manifest.txt", "w") as fh:
        fh.write("\n".join(total) + "\n")
    size = sum(os.path.getsize(os.path.join(ROOT, f)) for f in files)
    with open("/tmp/head_sweep_size.txt", "w") as fh:
        fh.write(f"{size / 1e9:.2f} GB\n")
    print(f"[manifest] {len(files)} fichiers embeddings ({size/1e9:.2f} GB) "
          f"+ {len(results)} résultats json locaux → /tmp/head_sweep_manifest.txt")
    if missing:
        print(f"[manifest] ATTENTION {len(missing)} fichiers introuvables localement :")
        for name, f in missing[:10]:
            print(f"  - {name}: {f}")
        print("  (le script push les signalera ; sur Narval, vérifier $SCRATCH)")


if __name__ == "__main__":
    main()
