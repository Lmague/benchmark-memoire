#!/usr/bin/env python3
"""Construit le split PILOTE (17 orthos du coords_manifest) — 10 train / 1 val / 6 test.

Décision figée AVANT toute extraction de features ou tout résultat (2026-10-31,
research/gnn_grid_2026-10-31/PRE_REGISTRATION.md §4).

Règles appliquées (déterministes, aucune donnée de performance consultée) :
  - val  = tussockfield51 : ortho ≥ 1 000 tuiles, meilleure couverture de classes parmi les
           orthos éligibles (7 classes), nécessaire à la sélection de C et à l'early stopping.
  - test = 6 orthos couvrant obligatoirement les classes rares ARCA(1), DRYI(3), RUBC(7)
           et les orthos de très faible effectif (lichentundra35/41/44) ; sélectionnés pour
           équilibrer le nombre de tuiles test (~41 % du total).
  - train = les 10 orthos restants ; doivent contenir les 11 classes.
Aucune tuile ne change d'orthomosaïque ; aucune arête inter-ortho n'est possible.

Usage : python code/make_pilot_split.py [--manifest ...] [--out ...]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os

TRAIN = [
    "20230724_alder25_m3m", "20230724_alder4_m3m", "20230724_birch26_m3m",
    "20230724_birch30_m3m", "20230724_birch54_m3m", "20230724_sedge_mosstundra34_m3m",
    "20230724_riparianshrub36_m3m", "20230725_sedge_mosstundra19_m3m",
    "20230725_tussockfield_m3m", "20230724_lichentundra35_m3m",
]
VAL = ["20230728_tussockfield51_m3m"]
TEST = [
    "20230724_sedge_mosstundra22_m3m", "20230725_birch2_m3m", "20230728_birch42_m3m",
    "20230724_riparianshrub23_m3m", "20230724_lichentundra41_m3m",
    "20230728_lichentundra44_m3m",
]
CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RUBC", "SEDG", "TUSS", "WILL"]


def rel_key(ctx_key: str) -> str:
    """'arctic_vegetation/<ortho>/<cls>/tile_x.png' -> même clé sous tiles/."""
    return ctx_key


def main() -> None:
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(os.path.dirname(root))
    ap.add_argument("--manifest", default=os.path.join(repo, "out/context/coords_manifest.json"))
    ap.add_argument("--out", default=os.path.join(root, "out/pilot_split.json"))
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    entries = man["entries"]
    orthos = sorted({v["ortho"] for v in entries.values()})
    assigned = TRAIN + VAL + TEST
    assert len(assigned) == len(set(assigned)) == 17, (len(assigned), len(set(assigned)))
    assert set(assigned) == set(orthos), set(orthos) ^ set(assigned)

    split_of = {o: "train" for o in TRAIN} | {o: "val" for o in VAL} | {o: "test" for o in TEST}
    rows = []
    for key in sorted(entries):
        e = entries[key]
        rows.append({
            "key": key,
            "ortho": e["ortho"],
            "class": e["class"],
            "label": CLASS_NAMES_11.index(e["class"]),
            "tile_row_off": e["tile_row_off"],
            "tile_col_off": e["tile_col_off"],
            "split": split_of[e["ortho"]],
        })

    # --- vérifications de couverture ---
    rep = {}
    for sp in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == sp]
        cnt = collections.Counter(r["label"] for r in sub)
        rep[sp] = {
            "n_tiles": len(sub),
            "n_orthos": len({r["ortho"] for r in sub}),
            "classes_present": sorted(cnt),
            "class_counts": {CLASS_NAMES_11[k]: v for k, v in sorted(cnt.items())},
        }
    for sp in ("train", "test"):
        missing = sorted(set(range(11)) - set(rep[sp]["classes_present"]))
        assert not missing, f"{sp} manque les classes {missing}"
        print(f"[OK] {sp}: 11 classes présentes")
    rare = {1: "ARCA", 3: "DRYI", 7: "RUBC"}
    for lab, name in rare.items():
        assert rep["test"]["class_counts"].get(name, 0) > 0, f"{name} absent du test"
    print("[OK] classes rares ARCA/DRYI/RUBC présentes dans le test")

    # --- intégrité de la grille : coordonnées uniques par ortho ---
    seen = collections.Counter((r["ortho"], r["tile_row_off"], r["tile_col_off"]) for r in rows)
    dup = [k for k, v in seen.items() if v > 1]
    assert not dup, f"coordonnées dupliquées: {dup[:5]}"

    payload = {
        "protocol": "pilot_17orthos_10_1_6",
        "source_manifest": os.path.abspath(a.manifest),
        "source_manifest_sha256": hashlib.sha256(open(a.manifest, "rb").read()).hexdigest(),
        "decision_date": "2026-10-31",
        "n_tiles": len(rows),
        "orthos": {"train": TRAIN, "val": VAL, "test": TEST},
        "report": rep,
        "tiles": rows,
    }
    body = json.dumps(payload, indent=1, sort_keys=True).encode()
    payload["sha256_without_hash"] = hashlib.sha256(body).hexdigest()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(json.dumps(rep, indent=1))
    print("hash:", payload["sha256_without_hash"])
    print("écrit:", a.out)


if __name__ == "__main__":
    main()
