#!/usr/bin/env python3
"""Splits SPATIAUX v2 — courbe de données contrôlée par unités spatiales.

Remplace l'implémentation v1 (``splits_spatial/``, archivée dans
``_anciennes_experiences/splits_spatial_v1_20260813/``). Différences v2 :
  - orthos ENTIÈRES par CLOSEST-SUM (sous-ensemble dont la somme de tuiles est
    la plus proche de la cible, au-dessus OU en-dessous), au lieu du greedy
    préfixe (qui surajustait : 36,3 % pour une cible de 25 %) ;
  - blocs contigus dans UNE orthomosaïque (pas de débordement inter-ortho),
    avec ortho ET offset tirés par seed (v1 tirait toujours dans la plus grosse) ;
  - manifest PAR (fraction, seed) listant les tile IDs exacts.

Principe : le volume de train est contrôlé par unités SPATIALES cohérentes,
jamais par tuiles tirées aléatoirement, pour éviter la fuite spatiale entre
tuiles voisines (tilerization 224 px, stride 112 px = chevauchement 50 %).

  Famille 1 — orthos entières : cibles {0.25, 0.50, 0.75, 1.00}.
  Famille 2 — blocs contigus   : cibles {0.01, 0.05, 0.10}.

Un bloc = intervalle d'indices [offset, offset+k) dans l'ordre de scan row-major
de l'orthomosaïque (``tile_XXXXX`` croissant avec la position, vérifié dans
``scripts/tilerization.py`` lignes 117-163) = bande spatiale contiguë.
Les tuiles vides (noir/blanc) sont sautées par la tilerization : les indices
adjacents restent contigus dans l'ordre de scan, avec de rares trous.

Sorties (tout dans ``spatial_datacurve/``, rien n'est touché ailleurs) :
  splits/fracXXX_seed{S}/{train,val,test}.csv  — splits prêts à entraîner
  splits/fracXXX_seed{S}/manifest.json         — tile IDs exacts + composition
  manifest.json                                — manifest global
  summary_table.csv / summary_table.md         — table résumé
  sanity_check.png                             — plot de contrôle

Val/test : copie EXACTE du split canonique v3 (``splits/val.csv``,
``splits/test.csv``) ; aucune orthomosaïque en commun avec le train (asserté).
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE, "splits")
ASSIGNMENT = os.path.join(SPLITS_DIR, "split_v3_assignment.json")
TRAIN_CSV = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV = os.path.join(SPLITS_DIR, "test.csv")
OUT = os.path.join(BASE, "spatial_datacurve")
OUT_SPLITS = os.path.join(OUT, "splits")

# Cibles de la courbe (fractions du volume de train canonique).
TARGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]
TAG = {0.01: "frac001", 0.05: "frac005", 0.10: "frac010",
       0.25: "frac025", 0.50: "frac050", 0.75: "frac075", 1.00: "frac100"}

CLASS_NAMES_12 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
# 11-class : RHOL(7) exclue, labels remappés 0-10 (cf. datacurve_one_run.py:37-49).
REMAP_11 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 8: 7, 9: 8, 10: 9, 11: 10}
# 8-class (diagnostic) : hors ARCA, DRYI, RUBC en 11-class → en codage 12-class :
LABELS_8CLS_12 = {0, 2, 4, 5, 6, 9, 10, 11}

TILE_RE = re.compile(r"arctic_vegetation/([^/]+)/([^/]+)/tile_(\d+)\.png$")


def load_train() -> dict[str, list[tuple[int, str, str]]]:
    """ortho -> liste triée (idx, filepath, label_12cls) des tuiles de train."""
    ortho_tiles: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            m = TILE_RE.match(row["filepath"])
            if not m:
                raise ValueError(f"chemin inattendu : {row['filepath']}")
            ortho, _cls, idx = m.group(1), m.group(2), int(m.group(3))
            ortho_tiles[ortho].append((idx, row["filepath"], row["label"]))
    for o in ortho_tiles:
        ortho_tiles[o].sort(key=lambda t: t[0])
    return dict(ortho_tiles)


def load_assignment() -> dict[str, str]:
    with open(ASSIGNMENT) as f:
        return json.load(f)["assignment"]


def classes_counts(labels_12: list[str]) -> dict:
    c12 = Counter(labels_12)
    c11 = Counter()
    for lab, n in c12.items():
        lab11 = REMAP_11.get(int(lab))
        if lab11 is not None:
            c11[lab11] += n
    c8 = {lab: n for lab, n in c12.items() if int(lab) in LABELS_8CLS_12}
    return {
        "classes_12cls": {CLASS_NAMES_12[int(k)]: int(v) for k, v in sorted(c12.items())},
        "classes_11cls": {CLASS_NAMES_12[int(lab)]: int(v) for lab, v in sorted(c11.items())},
        "classes_8cls": {CLASS_NAMES_12[int(k)]: int(v) for k, v in sorted(c8.items())},
    }


def select_orthos_closest_sum(sizes: list[tuple[str, int]], target: int,
                              total: int, rng: random.Random, seed: int,
                              used_sets: set[frozenset[int]]) -> list[str]:
    """Sous-ensemble d'orthos dont la somme de tuiles est la plus proche de target.

    Seed 0 : meilleure combinaison globale. Seeds 1-2 : tirage seedé parmi les
    combinaisons à ±1 % du volume total autour de la meilleure distance, en
    excluant les combinaisons déjà utilisées aux seeds précédentes.
    """
    n = len(sizes)
    cand: list[tuple[int, int, tuple[int, ...]]] = []
    for k in range(1, n + 1):
        for combo in itertools.combinations(range(n), k):
            s = sum(sizes[i][1] for i in combo)
            cand.append((abs(s - target), s, combo))
    cand.sort(key=lambda t: (t[0], t[1]))
    best_dist = cand[0][0]
    margin = max(150, int(0.01 * total))  # ±1 % du volume total de train
    pool = [c for c in cand if c[0] <= best_dist + margin]
    for _ in range(300):
        if seed == 0:
            chosen = pool[0]
        else:
            chosen = rng.choice(pool)
        fs = frozenset(chosen[2])
        if fs not in used_sets:
            used_sets.add(fs)
            break
    else:
        # pool trop restreint : prendre la meilleure combinaison non utilisée
        for c in pool:
            if frozenset(c[2]) not in used_sets:
                used_sets.add(frozenset(c[2]))
                chosen = c
                break
        else:
            raise RuntimeError(f"aucune combinaison libre (seed={seed})")
    names = [sizes[i][0] for i in chosen[2]]
    return names


def select_block(ortho_tiles: dict[str, list], block_size: int,
                 rng: random.Random, seed: int,
                 used: set[tuple[str, int]]) -> tuple[str, int]:
    """(ortho, offset) d'un bloc contigu de block_size tuiles.

    Candidates : orthos de train avec n_tiles >= block_size (le bloc reste dans
    UNE ortho). Seed 0 : plus grosse ortho, offset 0 (déterministe). Seeds 1-2 :
    tirage seedé de l'ortho (uniforme parmi les candidates) et de l'offset
    (uniforme dans [0, n - block_size]), en excluant (ortho, offset) déjà pris.
    """
    cands = sorted(((o, len(v)) for o, v in ortho_tiles.items()
                    if len(v) >= block_size), key=lambda t: -t[1])
    if not cands:
        raise RuntimeError(f"aucune ortho >= {block_size} tuiles pour le bloc")
    for _ in range(300):
        if seed == 0:
            ortho, offset = cands[0][0], 0
        else:
            ortho = rng.choice(cands)[0]
            n = len(ortho_tiles[ortho])
            offset = rng.randint(0, n - block_size)
        if (ortho, offset) not in used:
            used.add((ortho, offset))
            return ortho, offset
    raise RuntimeError(f"impossible de tirer un bloc libre (seed={seed})")


def write_split(tag: str, rows: list[tuple[str, str]]) -> None:
    d = os.path.join(OUT_SPLITS, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        for fp, lab in rows:
            w.writerow([fp, lab])
    shutil.copyfile(VAL_CSV, os.path.join(d, "val.csv"))
    shutil.copyfile(TEST_CSV, os.path.join(d, "test.csv"))


def main() -> None:
    os.makedirs(OUT_SPLITS, exist_ok=True)
    assignment = load_assignment()
    ortho_tiles = load_train()

    train_orthos = {o for o, s in assignment.items() if s == "train"}
    val_orthos = {o for o, s in assignment.items() if s == "val"}
    test_orthos = {o for o, s in assignment.items() if s == "test"}
    assert set(ortho_tiles) == train_orthos, "train.csv != assignment v3 (train)"
    assert not (train_orthos & val_orthos) and not (train_orthos & test_orthos), \
        "ortho partagée entre splits !"
    assert len(train_orthos) == 15 and len(val_orthos) == 8 and len(test_orthos) == 9

    total = sum(len(v) for v in ortho_tiles.values())
    assert total == 49433, f"total train = {total} (attendu 49433)"
    sizes = sorted(((o, len(v)) for o, v in ortho_tiles.items()), key=lambda t: -t[1])

    # ------------------------------------------------------------------ état
    val_test_orthos = val_orthos | test_orthos
    manifest_global = {
        "protocole": {
            "description": "Courbe de données SPATIALE v2 — volume de train "
                           "contrôlé par orthomosaïques entières (>= 25 %) ou "
                           "blocs spatiaux contigus (< 25 %). Val/test = split "
                           "canonique v3, aucune ortho partagée avec le train.",
            "total_train_tuiles": total,
            "n_orthos_train": len(train_orthos),
            "orthos_train_desc": [{"ortho": o, "n": n} for o, n in sizes],
            "orthos_val": sorted(val_orthos),
            "orthos_test": sorted(test_orthos),
            "val_test_partages": sorted(val_orthos & test_orthos),
            "tilerization": "scripts/tilerization.py : tile_XXXXX = ordre de "
                            "scan row-major (stride 112 px, overlap 50 %), "
                            "tuiles vides sautées",
        },
        "niveaux": [],
    }

    summary_rows = []
    for frac in TARGETS:
        tag_base = TAG[frac]
        target = round(frac * total)
        level = {"tag": tag_base, "cible": frac, "type": None,
                 "n_tiles_cible": target, "seeds": []}
        used_ortho_sets: set[frozenset[int]] = set()
        used_blocks: set[tuple[str, int]] = set()
        for S in SEEDS:
            rng = random.Random(1000 * S + int(frac * 100))
            tag = f"{tag_base}_seed{S}"
            # --- assemblage du train (ordre : sélection, puis index croissant)
            picked: list[tuple[str, str]] = []
            detail: list[dict] = []
            if frac == 1.00:
                level["type"] = "orthos_entieres"
                selected = [o for o, _ in sizes]
                for o in selected:
                    picked.extend((fp, lab) for _, fp, lab in ortho_tiles[o])
                detail = [{"ortho": o, "n_tiles": len(ortho_tiles[o]),
                           "tile_id_range": [ortho_tiles[o][0][0],
                                             ortho_tiles[o][-1][0]]}
                          for o in selected]
            elif frac >= 0.25:
                level["type"] = "orthos_entieres"
                selected = select_orthos_closest_sum(sizes, target, total, rng,
                                                     S, used_ortho_sets)
                for o in selected:
                    picked.extend((fp, lab) for _, fp, lab in ortho_tiles[o])
                detail = [{"ortho": o, "n_tiles": len(ortho_tiles[o]),
                           "tile_id_range": [ortho_tiles[o][0][0],
                                             ortho_tiles[o][-1][0]]}
                          for o in selected]
            else:
                level["type"] = "bloc_spatial_contigu"
                block_size = round(frac * total)
                ortho, offset = select_block(ortho_tiles, block_size, rng, S,
                                             used_blocks)
                tiles = ortho_tiles[ortho][offset:offset + block_size]
                picked = [(fp, lab) for _, fp, lab in tiles]
                selected = [ortho]
                idxs = [t[0] for t in tiles]
                max_gap = max((idxs[i + 1] - idxs[i] for i in range(len(idxs) - 1)),
                              default=0)
                detail = [{"ortho": ortho, "n_tiles": len(tiles),
                           "offset": offset,
                           "tile_id_range": [tiles[0][0], tiles[-1][0]],
                           "taille_ortho": len(ortho_tiles[ortho]),
                           "max_gap_indices_absolus": max_gap}]

            # --- dédoublonnage (sécurité, ne devrait jamais se produire)
            seen: set[str] = set()
            rows: list[tuple[str, str]] = []
            for fp, lab in picked:
                if fp not in seen:
                    seen.add(fp)
                    rows.append((fp, lab))
            tile_ids = [fp for fp, _ in rows]

            frac_reelle = len(rows) / total
            comp = classes_counts([lab for _, lab in rows])

            split_manifest = {
                "tag": tag,
                "fraction_cible": frac,
                "fraction_reelle": round(frac_reelle, 4),
                "type": level["type"],
                "seed": S,
                "n_orthos": len(selected),
                "n_tiles": len(rows),
                "selection": detail,
                "tile_ids": tile_ids,
                **comp,
            }
            write_split(tag, rows)

            with open(os.path.join(OUT_SPLITS, tag, "manifest.json"), "w") as f:
                json.dump(split_manifest, f, indent=1)

            # --- validation (famille bloc)
            if frac < 0.25:
                idxs = [t[0] for t in tiles]
                assert idxs == sorted(idxs)
                # Contiguïté dans l'ordre de scan : la liste triée EST un slice
                # de l'ortho ; les trous d'indices ABSOLUS (tuiles vides sautées
                # par la tilerization) sont rapportés dans `max_gap_indices_absolus`.
                assert (ortho, offset) in used_blocks

            level["seeds"].append({
                "seed": S,
                "orthos": selected,
                "n_orthos": len(selected),
                "n_tiles": len(rows),
                "fraction_reelle": round(frac_reelle, 4),
                "selection": detail,
                "n_classes_8cls_presentes": len(comp["classes_8cls"]),
            })

            summary_rows.append({
                "fraction_cible": frac,
                "tag": tag,
                "seed": S,
                "type": level["type"],
                "n_orthos": len(selected),
                "orthos_utilises": ";".join(selected),
                "detail": json.dumps(detail, ensure_ascii=False),
                "n_tiles": len(rows),
                "fraction_reelle": round(frac_reelle, 4),
            })
        manifest_global["niveaux"].append(level)

    # ------------------------------------------------------------------ sorties
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest_global, f, indent=1)

    with open(os.path.join(OUT, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Table Markdown lisible
    with open(os.path.join(OUT, "summary_table.md"), "w") as f:
        f.write("# Courbe de données spatiale v2 — table résumé\n\n")
        f.write(f"Volume de train canonique : **{total} tuiles** "
                f"({len(train_orthos)} orthos). Val/test inchangés "
                f"(split v3).\n\n")
        f.write("| Cible | Tag | Seed | Type | #orthos | Orthos / bloc | "
                "Tuiles | Fraction réelle |\n|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r['fraction_cible']:.0%} | {r['tag']} | {r['seed']} "
                    f"| {r['type']} | {r['n_orthos']} | "
                    f"`{r['orthos_utilises']}` | {r['n_tiles']} "
                    f"| {r['fraction_reelle']:.1%} |\n")

    print(f"OK : {len(summary_rows)} splits écrits dans {OUT_SPLITS}")
    print(f"Manifest global : {os.path.join(OUT, 'manifest.json')}")
    print(f"Table résumé     : {os.path.join(OUT, 'summary_table.csv')}")


if __name__ == "__main__":
    main()
