"""Garde-fous de reproductibilité (dépôt) — sans dépendance (stdlib uniquement).

Vérifie les invariants qui, s'ils cassent, font dériver les chiffres canoniques sans
qu'aucun test ne s'en aperçoive :

1. la grille C par défaut de `configs/base.yaml` est la grille canonique à 6 points
   (la grille restreinte à 4 points produit les valeurs 0,4675/0,4680 dépréciées) ;
2. les points d'entrée canoniques (`probe.py`, `scripts/run_pipeline.py`) forcent le
   BLAS mono-thread AVANT tout import de `src` (donc de numpy) ;
3. le `Makefile` exporte les variables de threads pour la cible `paper` ;
4. les valeurs interdites (0,4675 / 0,4680 / 0,5256) n'apparaissent pas dans les
   sources actives du manuscrit.

Exécution : ``python3 tests/test_repo_guards.py`` (ou via pytest).
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
CANONICAL_GRID = [1.0e-4, 1.0e-3, 1e-2, 1e-1, 1.0, 10.0]


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def test_base_yaml_has_canonical_c_grid():
    txt = _read("configs", "base.yaml")
    m = re.search(r"C_grid:\s*\[([^\]]*)\]", txt)
    assert m, "C_grid absent de configs/base.yaml"
    vals = [float(x) for x in m.group(1).split(",")]
    assert len(vals) == 6, f"grille C à {len(vals)} points au lieu de 6"
    assert vals == CANONICAL_GRID, f"grille C non canonique : {vals}"


def test_entrypoints_force_single_thread_before_src():
    for rel in ("probe.py", "scripts/run_pipeline.py"):
        txt = _read(*rel.split("/"))
        assert all(v in txt for v in THREAD_VARS), f"{rel} ne force pas les threads"
        i_thread = txt.index("OMP_NUM_THREADS")
        i_src = txt.index("from src") if "from src" in txt else len(txt)
        assert i_thread < i_src, f"{rel} : threads fixés après l'import de src/numpy"


def test_makefile_exports_threads():
    txt = _read("Makefile")
    for v in THREAD_VARS:
        assert v in txt, f"Makefile : {v} non exporté"


def test_forbidden_values_absent_from_manuscript():
    txt = _read("paper_arctic_fm_benchmark", "paper_arctic_fm_benchmark.tex")
    for val in ("0.4675", "0.4680", "0.5256", "0,4675", "0,4680", "0,5256"):
        assert val not in txt, f"valeur dépréciée {val} présente dans le manuscrit"


if __name__ == "__main__":
    for fn in (test_base_yaml_has_canonical_c_grid,
               test_entrypoints_force_single_thread_before_src,
               test_makefile_exports_threads,
               test_forbidden_values_absent_from_manuscript):
        fn()
        print(f"[OK] {fn.__name__}")
    print("Garde-fous : 4/4")
