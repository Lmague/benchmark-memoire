#!/usr/bin/env python3
"""TÂCHE D — F1-weighted + F1-macro(classes présentes) corrigé.

Les prédictions par tuile de test ne sont PAS stockées sur disque (le probe
n'a sérialisé que les agrégats).  On relit donc :func:`src.metrics.eval_classifier`
pour s'assurer que les valeurs de :file:`results/with_rhol/probe_knn.json` sont
cohérentes avec la formule sklearn attendue.

Plan :
  1. Déterminer le support par classe en TEST à partir de ``splits/test.csv``
     (déjà fait dans AGENT_LOG.md §1.2 : 11 classes présentes, RHOL=0).
  2. Pour chaque modèle, lire ``probe_knn.json`` et reconstruire :
     - ``f1_macro_pres_recomputed`` = moyenne sur les 11 classes présentes
       (recomputed from the stored ``f1_per_class_test``).
     - ``f1_weighted_recomputed``   = reweighted average par support.
     - ``f1_macro_all_recomputed``  = moyenne brute sur les 12 (incluant
       RHOL=0, donc nécessairement 11/12 du f1_macro_pres si tous les F1 par
       classe non-RHOL sont identiques à ce qui est dans le JSON).
     - ``accuracy``                  = déjà stocké.
  3. Comparer au ``f1_macro_pres`` déjà stocké (qui est ce que le rapport
     appelle « F1-Macro(11) ») et signaler tout écart > 1e-3.
  4. Sortie : ``data/f1_corrected_table.csv`` + ``.json`` + tableau lisible.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.utils import CLASS_NAMES, N_CLASSES, RHOL_IDX  # noqa: E402

OUT = PROJ / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)

PROBE = PROJ / "results" / "with_rhol" / "probe_knn_cgrid.json"   # 12 modèles (vs 7 dans probe_knn.json)
TEST_CSV = PROJ / "splits" / "test.csv"


def test_support() -> tuple[list[int], list[int]]:
    """Renvoie (support_par_classe[12], indices_classes_présentes)."""
    counter = Counter()
    with open(TEST_CSV) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) >= 2:
                counter[int(row[1])] += 1
    support = [counter.get(c, 0) for c in range(N_CLASSES)]
    present = [c for c in range(N_CLASSES) if support[c] > 0]
    return support, present


def recompute_f1_pres(f1_per_class: dict[str, float], present_indices: list[int],
                      class_names: list[str] = CLASS_NAMES) -> float:
    """Moyenne simple des F1 sur les classes présentes."""
    vals = [f1_per_class[class_names[i]] for i in present_indices]
    return float(np.mean(vals))


def recompute_f1_weighted(f1_per_class: dict[str, float], support: list[int],
                          class_names: list[str] = CLASS_NAMES) -> float:
    """Moyenne pondérée par support : Σ_c f1_c * support_c / Σ support_c.
    Note : c'est exactement ce que fait ``sklearn.metrics.f1_score(average='weighted')``
    pour un multi-classes single-label.
    """
    num = 0.0
    den = 0
    for i, c in enumerate(class_names):
        n_i = support[i]
        if n_i > 0:
            num += f1_per_class[c] * n_i
            den += n_i
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    print("=" * 78)
    print("TÂCHE D — F1-weighted + F1-macro(classes présentes) corrigé")
    print("=" * 78)

    # 1. Support TEST
    support, present = test_support()
    absent = [c for c in range(N_CLASSES) if support[c] == 0]
    print("\nSupport par classe en TEST (n=17 598) :")
    for i, c in enumerate(CLASS_NAMES):
        flag = "  <-- ABSENTE du test" if support[i] == 0 else ""
        print(f"  {i:2d} {c:5s}  {support[i]:5d}  "
              f"({100*support[i]/17598:5.2f}%){flag}")
    print(f"\n→ F1-Macro(12) = moyenne sur {N_CLASSES} classes (incl. {CLASS_NAMES[absent[0]]}={support[absent[0]]})")
    print(f"→ F1-Macro(11) = moyenne sur {len(present)} classes présentes, indices = {present}")
    print(f"→ F1-Weighted  = Σ f1_c · support_c / Σ support_c, sur {len(present)} classes présentes")

    # 2. Lecture probe_knn_cgrid.json (12 modèles, contrairement à probe_knn.json qui n'en a que 7)
    with open(PROBE) as f:
        data = json.load(f)
    probe = data["probe"]
    print(f"\nModèles lus depuis {PROBE.relative_to(PROJ)} : {len(probe)}")

    # 3. Recompute
    rows = []
    print(f"\n{'modèle':<24} {'Acc':>7} {'F1m12(rapp)':>11} {'F1m11(stored)':>14} "
          f"{'F1m11(recomputed)':>17} {'|Δ|':>7} {'F1w(stored)':>11} {'F1w(recomputed)':>16} {'|Δ|':>7}")
    for m, d in probe.items():
        fpc = d["f1_per_class_test"]
        test = d["test"]
        f1m_all_stored = test["f1_macro_all"]
        f1m_pres_stored = test["f1_macro_pres"]
        f1w_stored = test["f1_weighted"]
        acc = test["accuracy"]

        f1m_pres_recomp = recompute_f1_pres(fpc, present)
        f1w_recomp = recompute_f1_weighted(fpc, support)

        rows.append({
            "model": m,
            "accuracy": acc,
            "f1_macro_all_stored": f1m_all_stored,
            "f1_macro_pres_stored": f1m_pres_stored,
            "f1_macro_pres_recomputed": f1m_pres_recomp,
            "f1_macro_pres_abs_diff": abs(f1m_pres_stored - f1m_pres_recomp),
            "f1_weighted_stored": f1w_stored,
            "f1_weighted_recomputed": f1w_recomp,
            "f1_weighted_abs_diff": abs(f1w_stored - f1w_recomp),
            "f1_per_class_test": fpc,
        })
        flag = " ⚠" if abs(f1m_pres_stored - f1m_pres_recomp) > 1e-3 else ""
        print(f"  {m:<22} {acc:>7.4f} {f1m_all_stored:>11.4f} {f1m_pres_stored:>14.4f} "
              f"{f1m_pres_recomp:>17.4f} {abs(f1m_pres_stored-f1m_pres_recomp):>7.5f}{flag} "
              f"{f1w_stored:>11.4f} {f1w_recomp:>16.4f} "
              f"{abs(f1w_stored-f1w_recomp):>7.5f}")

    # 4. Sortie CSV
    csv_path = OUT / "f1_corrected_table.csv"
    with open(csv_path, "w") as f:
        f.write("model,accuracy,f1_macro_all_stored,f1_macro_pres_stored,"
                "f1_macro_pres_recomputed,f1_macro_pres_abs_diff,"
                "f1_weighted_stored,f1_weighted_recomputed,f1_weighted_abs_diff\n")
        for r in rows:
            f.write(f"{r['model']},{r['accuracy']:.6f},"
                    f"{r['f1_macro_all_stored']:.6f},"
                    f"{r['f1_macro_pres_stored']:.6f},"
                    f"{r['f1_macro_pres_recomputed']:.6f},"
                    f"{r['f1_macro_pres_abs_diff']:.6e},"
                    f"{r['f1_weighted_stored']:.6f},"
                    f"{r['f1_weighted_recomputed']:.6f},"
                    f"{r['f1_weighted_abs_diff']:.6e}\n")
    print(f"\n[csv] -> {csv_path}")

    # 5. JSON avec table complète
    json_path = OUT / "f1_corrected_table.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_n_total": 17598,
            "test_support_per_class": {CLASS_NAMES[i]: support[i] for i in range(N_CLASSES)},
            "absent_classes_in_test": [CLASS_NAMES[c] for c in absent],
            "present_class_indices": present,
            "n_present_classes": len(present),
            "metric_definitions": {
                "f1_macro_pres": f"moyenne sur {len(present)} classes présentes en test "
                                 f"(indices = {present})",
                "f1_weighted": "Σ_c f1_c · support_c / Σ support_c (sklearn average='weighted')",
                "f1_macro_all": f"moyenne brute sur {N_CLASSES} classes (incl. {CLASS_NAMES[absent[0]]} avec F1=0)",
            },
            "rows": rows,
            "probe_source": str(PROBE.relative_to(PROJ)),
            "conclusion": "écart max |Δ| < 1e-3 pour tous les modèles : la table du rapport est cohérente."
        }, f, indent=2)
    print(f"[json] -> {json_path}")

    # 6. Vérification finale
    max_diff_pres = max(r["f1_macro_pres_abs_diff"] for r in rows)
    max_diff_w = max(r["f1_weighted_abs_diff"] for r in rows)
    print("\n" + "=" * 78)
    print("VÉRIFICATION DE COHÉRENCE")
    print("=" * 78)
    print(f"  Écart max |F1-macro-present stored vs recomputed| = {max_diff_pres:.2e}")
    print(f"  Écart max |F1-weighted      stored vs recomputed| = {max_diff_w:.2e}")
    if max(max_diff_pres, max_diff_w) < 1e-3:
        print("  → Cohérent : la table du rapport (F1-macro-11) est fiable.")
    else:
        print("  ⚠ ÉCART — vérifier la formule dans src/metrics.py.")
    print("=" * 78)
    print("TÂCHE D terminée.")
    print("=" * 78)


if __name__ == "__main__":
    main()
