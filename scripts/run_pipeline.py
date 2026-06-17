#!/usr/bin/env python3
"""Orchestration canonique du pipeline A→F (passe nocturne 2).

Exécute séquentiellement :
  A. probe         → results/with_rhol/probe_knn_cgrid.json   (12 modèles, grille étendue, best_C par val)
  B. latent        → results/with_rhol/latent_metrics.json + results/geometry_extended_12models.json
  C. significance  → results/significance_matrix_all12.json   (bootstrap apparié n=1000)
  D. transfer      → results/transfer/{spectrum_metrics,logme_scores,correlations}.{csv,json}
                     + results/correlations.json              (n=9, n=12, top-8, top-6)
                     + results/with_rhol/latent_metrics.json récapitulatif
                     + results/transfer/headline_pairs_paired_tests.{json,csv}
                     + results/transfer/aso_matrix_eps_min.{json,csv}
  E. figures       → docs/figures/*.png, results/figures_paper/{headline_pair,controlled_pair,...}.png
  F. tables        → results/transfer/{tab_spectrum,tab_correl,tab_palier,tab_logme,tab_paired,tab_f1}.tex

Garanties :
  * déterministe (seed=42 partout, les 6 tâches)
  * idempotent (peut être ré-exécuté sans état caché)
  * portable (lit ``cfg.paths.emb_dir`` / ``results_dir``)
  * aucun chemin absolu (sauf via configs/base.yaml:paths_local)

Usage :
  python scripts/run_pipeline.py                  # exécute A→F
  python scripts/run_pipeline.py --only A C       # exécute A et C seulement
  python scripts/run_pipeline.py --skip E F       # saute E et F
  python scripts/run_pipeline.py --n-bootstrap 200  # override n (utile en CI)

Pré-requis :
  * Les 12 modèles d'embeddings dans ``embeddings/`` (lecture seule, NE PAS MODIFIER).
  * ``configs/frozen_eval.yaml`` pointe sur ``include_finetuned: true`` (pour 12 modèles).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _run(label: str, cmd: list[str], log_file: Path | None = None) -> int:
    """Exécute ``cmd`` (sous-process) avec capture et log."""
    _log(f"--- {label} ---\n    $ {' '.join(cmd)}")
    t0 = time.time()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as lf:
            lf.write(f"\n\n=== {label} | {' '.join(cmd)} ===\n")
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=PROJ)
    else:
        rc = subprocess.run(cmd, cwd=PROJ)
    dt = time.time() - t0
    _log(f"    → rc={rc.returncode}  ({dt:.1f}s)")
    return rc.returncode


# ------------------------------------------------------------------ A. probe
def task_A(args, log_file: Path) -> int:
    """Régénère probe_knn_cgrid.json pour les 12 modèles (grille étendue C∈{1e-4..10}).
    Best_C sélectionné sur val par f1_macro_all (méthodo canonique).

    Le nom de sortie ``probe_knn_cgrid.json`` (via --output-tag cgrid) est ce que
    lisent TOUS les scripts en aval (task_a/b/c/d, compute_correlations, figures).
    """
    cfg = "configs/benchmark_12models.yaml"
    cmd = ["python3", "probe.py", "--config", cfg, "--output-tag", "cgrid"]
    return _run("A. probe (12 modèles, grille étendue, best_C par val)", cmd, log_file)


# ------------------------------------------------------------------ B. latent
def task_B(args, log_file: Path) -> int:
    """Métriques géométriques de l'espace latent — passe with_rhol (12 classes, RHOL=0)."""
    rc = _run("B.1 latent — RankMe + anisotropie (12 modèles, 11 classes)",
              ["python3", "analyze.py", "--config", "configs/benchmark_12models.yaml"],
              log_file)
    if rc != 0:
        return rc
    return _run("B.2 geometry_extended_12models (stable_rank, TwoNN, Fisher, kNN purity)",
                ["python3", "scripts/geometry_extended_12models.py"], log_file)


# ------------------------------------------------------------------ C. significance
def task_C(args, log_file: Path) -> int:
    """Bootstrap apparié n=1000 (seed=42) sur les 12 modèles + 3 tests appariés
    par tuile (ASO + bootstrap + permutation) sur les paires palier A.

    Idempotent : si les artefacts existent déjà, skip (sauf si --force).
    task_c_b_paired.py prend ~30 min et n'est utile qu'à régénérer."""
    n = int(getattr(args, "n_bootstrap", 1000))
    sig_json = PROJ / "results" / "significance_matrix_all12.json"
    paired_json = PROJ / "results" / "transfer" / "headline_pairs_paired_tests.json"
    force = bool(getattr(args, "force", False))

    if not force:
        if sig_json.exists() and paired_json.exists():
            print(f"[run_pipeline] C outputs déjà présents → SKIP (utiliser --force pour re-calculer).")
            return 0

    rc = _run(f"C.1 bootstrap apparié n={n} — 12 modèles, f1_macro_pres",
              ["python3", "scripts/significance_matrix.py",
               "--probe-json", "results/with_rhol/probe_knn_cgrid.json",
               "--n-bootstrap", str(n),
               "--output-json", "results/significance_matrix_all12.json",
               "--output-png", "results/significance_matrix_all12.png"],
              log_file)
    if rc != 0:
        return rc
    return _run("C.2 tests appariés par tuile — paires palier A (best_C canonique)",
                ["python3", "scripts/task_c_b_paired.py"], log_file)


# ------------------------------------------------------------------ D. transfer
def task_D(args, log_file: Path) -> int:
    """Transferability (LogME + α-ReQ + NESum) sur les 12 modèles, plus corrélations
    géométrie↔F1 sur n=9 (frozen) et paliers top-K compétitifs."""
    rc = _run("D.1 spectrum (α-ReQ + NESum) — 12 modèles, 20k subsample test, seed 42",
              ["python3", "scripts/task_a_spectrum.py"], log_file)
    if rc != 0:
        return rc
    rc = _run("D.2 LogME — 12 modèles, embeddings train 49 433 tuiles",
              ["python3", "scripts/task_b_logme.py"], log_file)
    if rc != 0:
        return rc
    return _run("D.3 corrélations géométrie↔F1 (n=9 frozen + n=12 + paliers top-8/6)",
                ["python3", "scripts/compute_correlations.py",
                 "--config", "configs/frozen_eval.yaml",
                 "--probe-json", "results/with_rhol/probe_knn_cgrid.json",
                 "--latent-json", "results/with_rhol/latent_metrics.json",
                 "--probe-json-all", "results/with_rhol/probe_knn_cgrid.json",
                 "--output", "results/correlations.json",
                 "--n-bootstrap", "1000",
                 "--fig", "docs/figures/logme_vs_f1.png"],
                log_file)


# ------------------------------------------------------------------ E. figures
def task_E(args, log_file: Path) -> int:
    """Régénère les figures canoniques (paper + support) à partir des JSON produits."""
    rc = _run("E.1 figures paper (significance_holm, corr_forest, "
              "geometry_extended_scatter, controlled_pair, headline_pair)",
              ["python3", "scripts/make_paper_figures.py",
               "--sig", "results/significance_matrix.json",
               "--corrected", "results/significance_corrected.json",
               "--geom", "results/geometry_extended.json",
               "--sig12", "results/significance_matrix_all12.json",
               "--outdir", "results/figures_paper"],
              log_file)
    if rc != 0:
        return rc
    return _run("E.2 figures annexes (12 modèles, transfert, géométrie)",
                ["python3", "scripts/regenerate_all_figures_12.py"],
                log_file)


# ------------------------------------------------------------------ F. tables
def task_F(args, log_file: Path) -> int:
    """Génère les fragments LaTeX (tab_*.tex) depuis les CSV/JSON canoniques."""
    return _run("F. tables LaTeX depuis les CSV/JSON canoniques",
                ["python3", "scripts/gen_latex_tables.py"],
                log_file)


TASKS = {"A": task_A, "B": task_B, "C": task_C, "D": task_D, "E": task_E, "F": task_F}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    help="exécute UNIQUEMENT ces tâches (ex: --only A C F)")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="saute ces tâches (ex: --skip E F)")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="override du nombre de tirages bootstrap (défaut: 1000)")
    ap.add_argument("--force", action="store_true",
                    help="force le re-calcul de TOUTES les tâches (sans skip par défaut)")
    ap.add_argument("--log", default="results/transfer/pipeline_run.log",
                    help="fichier log de l'orchestration")
    args = ap.parse_args()

    log_path = PROJ / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # truncate
    log_path.write_text(f"# Pipeline A→F — démarré {_now()}\n")

    only = set(args.only or TASKS.keys())
    skip = set(args.skip or [])
    order = ["A", "B", "C", "D", "E", "F"]
    selected = [k for k in order if k in only and k not in skip]

    _log(f"Tâches sélectionnées : {' '.join(selected)}")
    t_global = time.time()
    failed = []
    for k in selected:
        rc = TASKS[k](args, log_path)
        if rc != 0:
            _log(f"!!! Tâche {k} a échoué (rc={rc}) — arrêt du pipeline.")
            failed.append((k, rc))
            break
    dt = time.time() - t_global
    _log(f"=== Pipeline terminé en {dt:.1f}s | failed={failed} ===")

    # Petit récapitulatif JSON pour le rapport
    summary = {
        "tasks_run": selected,
        "tasks_failed": failed,
        "elapsed_s": round(dt, 1),
        "log": str(log_path.relative_to(PROJ)),
    }
    (PROJ / "results" / "transfer" / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
