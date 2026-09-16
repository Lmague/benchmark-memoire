"""Tests de scripts/generate_splits_11cls.py sur un exemple synthétique.

Compatible pytest (fonctions ``test_*``) et exécution directe
(``python3 tests/test_generate_splits_11cls.py``) — aucune dépendance à pytest
lui-même, seulement numpy (déjà requis par le projet).
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

_SPEC = importlib.util.spec_from_file_location(
    "generate_splits_11cls", os.path.join(_REPO_ROOT, "scripts", "generate_splits_11cls.py")
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


def _write_csv(path: str, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows)


def test_remap_removes_rhol_and_closes_gap():
    # Labels bruts 0..11 (12 classes), RHOL=7, + une ligne RHOL supplémentaire.
    rows = [(f"tile_{i}.png", i) for i in range(12)] + [("tile_extra_rhol.png", 7)]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "train.csv")
        out_path = os.path.join(tmp, "train_11cls.csv")
        _write_csv(in_path, rows)

        n_before, n_after, labels_before, labels_after = _mod.remap_split(in_path, out_path)

        assert n_before == 13
        assert n_after == 11  # 2 lignes RHOL (label 7) retirées

        uniq_after = sorted(int(x) for x in np.unique(labels_after))
        assert uniq_after == list(range(11)), "labels 0-10 sans trou attendus"
        assert max(uniq_after) <= 10

        with open(out_path) as f:
            r = csv.reader(f)
            header = next(r)
            written_labels = [int(row[1]) for row in r]
        assert header == ["filepath", "label"]
        assert sorted(written_labels) == sorted(labels_after)


def test_remap_preserves_labels_below_rhol():
    rows = [("a.png", 0), ("b.png", 3), ("c.png", 6)]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "val.csv")
        out_path = os.path.join(tmp, "val_11cls.csv")
        _write_csv(in_path, rows)
        _, _, _, labels_after = _mod.remap_split(in_path, out_path)
        assert labels_after == [0, 3, 6]


def test_remap_shifts_labels_above_rhol_down_by_one():
    rows = [("a.png", 8), ("b.png", 11)]  # RUBC, WILL bruts (12cls)
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "test.csv")
        out_path = os.path.join(tmp, "test_11cls.csv")
        _write_csv(in_path, rows)
        _, _, _, labels_after = _mod.remap_split(in_path, out_path)
        assert labels_after == [7, 10]


def test_remap_all_rhol_yields_empty_output():
    rows = [("a.png", 7), ("b.png", 7)]
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "train.csv")
        out_path = os.path.join(tmp, "train_11cls.csv")
        _write_csv(in_path, rows)
        n_before, n_after, _, labels_after = _mod.remap_split(in_path, out_path)
        assert n_before == 2
        assert n_after == 0
        assert labels_after == []


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(_TESTS)} tests passed.")
