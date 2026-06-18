#!/usr/bin/env python3
"""Data curve Q5 — agrégation des résultats et figures.

À exécuter LOCALEMENT après rapatriement des résultats depuis Narval :

    python scripts/datacurve_aggregate.py \
        --runs-dir outputs/datacurve/runs \
        --emb-dir  outputs/datacurve/embeddings \
        --out-dir  outputs/datacurve

Sorties :
  outputs/datacurve/results_raw.csv           — un run par ligne
  outputs/datacurve/results_agg.csv           — agrégé par proportion
  outputs/datacurve/baselines.csv             — 4 modèles gelés × 2 métriques
  outputs/datacurve/learning_curve_pres.png   — courbe f1_macro_pres (11 cls)
  outputs/datacurve/learning_curve_8cls.png   — courbe f1_macro_8cls  (8 cls)
  outputs/datacurve/per_class_curve.png       — F1 par classe (Tier 2, si dispo)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

# Classes pour la métrique 8-cls (hors ARCA=1, DRYI=3, RHOL=7, RUBC=8)
LABELS_8CLS = [0, 2, 4, 5, 6, 9, 10, 11]   # ALDE BIRC LICH MOSS PETF SEDG TUSS WILL

# Baselines gelées à évaluer (lecture depuis embeddings/ local)
BASELINE_MODELS = [
    "dinov3_vitl16_lvd",   # meilleur gelé ViT-L — cible principale
    "dinov3_vitb16_lvd",   # gelé ViT-B — comparaison équitable
    "simdinov2_vitb16",    # SimDINOv2 ViT-B
    "simdinov2_vitl16",    # SimDINOv2 ViT-L
]

# Étiquettes lisibles pour la figure
_BASELINE_LABELS = {
    "dinov3_vitl16_lvd": "DINOv3 ViT-L LVD (gelé)",
    "dinov3_vitb16_lvd": "DINOv3 ViT-B LVD (gelé)",
    "simdinov2_vitb16":  "SimDINOv2 ViT-B (gelé)",
    "simdinov2_vitl16":  "SimDINOv2 ViT-L (gelé)",
}

# Couleurs des baselines (la cible principale est mise en avant)
_BASELINE_COLORS = {
    "dinov3_vitl16_lvd": "#e41a1c",   # rouge vif — cible principale
    "dinov3_vitb16_lvd": "#ff7f00",   # orange
    "simdinov2_vitb16":  "#4dac26",   # vert
    "simdinov2_vitl16":  "#984ea3",   # violet
}
_BASELINE_LSTYLE = {
    "dinov3_vitl16_lvd": "--",   # tirets longs
    "dinov3_vitb16_lvd": "-.",
    "simdinov2_vitb16":  ":",
    "simdinov2_vitl16":  (0, (5, 2, 1, 2)),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _probe_baseline(emb_dir: str, model_key: str,
                    C_grid=(0.001, 0.01, 0.1, 1.0, 10.0),
                    max_iter: int = 2000) -> dict:
    """Charge les embeddings gelés (train/val/test) et retourne les deux métriques."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from src.utils import make_canonical_lr

    E_tr = np.load(os.path.join(emb_dir, f"{model_key}_train.npy")).astype(np.float32)
    L_tr = np.load(os.path.join(emb_dir, f"{model_key}_train_labels.npy")).astype(np.int64)
    E_va = np.load(os.path.join(emb_dir, f"{model_key}_val.npy")).astype(np.float32)
    L_va = np.load(os.path.join(emb_dir, f"{model_key}_val_labels.npy")).astype(np.int64)
    E_te = np.load(os.path.join(emb_dir, f"{model_key}_test.npy")).astype(np.float32)
    L_te = np.load(os.path.join(emb_dir, f"{model_key}_test_labels.npy")).astype(np.int64)

    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr)
    X_va = sc.transform(E_va)
    X_te = sc.transform(E_te)

    # Sélection C sur val (f1_macro_all 12 classes, convention benchmark)
    best_c, best_f1v = C_grid[0], -1.0
    for c in C_grid:
        clf = make_canonical_lr(C=c, max_iter=max_iter)
        clf.fit(X_tr, L_tr)
        f1v = f1_score(L_va, clf.predict(X_va), average="macro",
                       labels=list(range(12)), zero_division=0)
        if f1v > best_f1v:
            best_c, best_f1v = c, f1v

    clf_final = make_canonical_lr(C=best_c, max_iter=max_iter)
    clf_final.fit(X_tr, L_tr)
    preds_te = clf_final.predict(X_te)

    # f1_macro_pres (11 cls) — RHOL absent du test → exclue naturellement
    present_te = sorted({int(v) for v in L_te})
    f1_pres = float(f1_score(L_te, preds_te, average="macro",
                             labels=present_te, zero_division=0))
    # f1_macro_8cls
    f1_8cls = float(f1_score(L_te, preds_te, average="macro",
                             labels=LABELS_8CLS, zero_division=0))

    print(f"  {model_key}: f1_pres={f1_pres:.4f}  f1_8cls={f1_8cls:.4f}"
          f"  best_C={best_c}", flush=True)
    return {"model": model_key, "f1_macro_pres_test": f1_pres,
            "f1_macro_8cls_test": f1_8cls, "best_C": best_c}


def _perclass_from_metrics(metrics_path: str) -> dict:
    """Lit f1_per_class_test depuis metrics.json (calculé par datacurve_one_run.py)."""
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        d = json.load(f)
    return d.get("f1_per_class_test", {})


def _interpolate_crossover(x_vals, y_mean, y_ref):
    """Retourne x interpolé où la courbe croise y_ref, ou None si jamais croisé."""
    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_mean, dtype=float)
    for i in range(len(y_arr) - 1):
        if (y_arr[i] < y_ref <= y_arr[i + 1]) or (y_arr[i] >= y_ref > y_arr[i + 1]):
            # Interpolation linéaire entre x[i] et x[i+1]
            t = (y_ref - y_arr[i]) / (y_arr[i + 1] - y_arr[i] + 1e-12)
            x_cross = x_arr[i] + t * (x_arr[i + 1] - x_arr[i])
            return int(round(x_cross))
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Lecture des runs
# ──────────────────────────────────────────────────────────────────────────────

def _read_runs(runs_dir: str) -> list[dict]:
    """Lit tous les metrics.json sous runs_dir/*/metrics.json."""
    rows = []
    if not os.path.isdir(runs_dir):
        return rows
    for entry in sorted(os.listdir(runs_dir)):
        m = os.path.join(runs_dir, entry, "metrics.json")
        if os.path.exists(m):
            with open(m) as f:
                d = json.load(f)
            rows.append(d)
            print(f"  {entry}: f1_pres={d.get('f1_macro_pres_test','?'):.4f}"
                  f"  f1_8cls={d.get('f1_macro_8cls_test','?'):.4f}", flush=True)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _make_learning_curve(
    agg: dict,                # {fraction: {n_train, f1_pres_mean, f1_pres_std, f1_8cls_mean, ...}}
    baselines: dict,          # {model_key: {f1_macro_pres_test, f1_macro_8cls_test}}
    metric: str,              # "pres" | "8cls"
    out_path: str,
    crossovers: dict,         # {model_key: n_tuiles ou None}
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_col = "f1_pres" if metric == "pres" else "f1_8cls"
    ylabel = ("F1-macro test (11 classes, sans RHOL)"
              if metric == "pres"
              else "F1-macro test (8 classes)")
    title = ("Courbe d'apprentissage — ViT-B/16 full fine-tuning\n"
             + ("Métrique primaire : F1-macro (11 classes)"
                if metric == "pres"
                else "Métrique diagnostique : F1-macro (8 classes)"))

    fracs_sorted = sorted(agg.keys())
    x = [agg[f]["n_train"] for f in fracs_sorted]
    y_mean = [agg[f][f"{metric_col}_mean"] for f in fracs_sorted]
    y_std = [agg[f][f"{metric_col}_std"] for f in fracs_sorted]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Courbe fine-tuned
    ax.semilogx(x, y_mean, "o-", color="#1f77b4", linewidth=2,
                markersize=6, label="ViT-B/16 full FT (mean ± std, 3 seeds)",
                zorder=5)
    ax.fill_between(x,
                    [m - s for m, s in zip(y_mean, y_std)],
                    [m + s for m, s in zip(y_mean, y_std)],
                    alpha=0.2, color="#1f77b4")

    # Lignes horizontales baselines
    for key in BASELINE_MODELS:
        if key not in baselines:
            continue
        val = baselines[key][f"f1_macro_{metric_col.replace('_', '_')}"]
        # Nommage cohérent
        val_key = "f1_macro_pres_test" if metric == "pres" else "f1_macro_8cls_test"
        val = baselines[key][val_key]
        is_main = key == "dinov3_vitl16_lvd"
        lw = 2.5 if is_main else 1.8
        lbl = _BASELINE_LABELS[key] + f" ({val:.4f})"
        ax.axhline(val, color=_BASELINE_COLORS[key],
                   linestyle=_BASELINE_LSTYLE[key], linewidth=lw,
                   label=lbl, zorder=3)
        # Annotation croisement
        cross = crossovers.get(key)
        if cross is not None:
            ax.axvline(cross, color=_BASELINE_COLORS[key],
                       linestyle=":", alpha=0.5, linewidth=1.2)
            ax.text(cross, ax.get_ylim()[0] + 0.001,
                    f"×{cross:,}", color=_BASELINE_COLORS[key],
                    fontsize=7.5, rotation=90, va="bottom", ha="right")
        elif len(x) > 0 and y_mean[-1] < val:
            # Jamais atteint : annotation
            ax.text(x[-1] * 1.05, val + 0.002,
                    "jamais croisé\n(100%)", color=_BASELINE_COLORS[key],
                    fontsize=7, va="bottom", ha="left")

    ax.set_xlabel("Tuiles d'entraînement (échelle log)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(which="both", alpha=0.3)
    ax.set_xlim(left=x[0] * 0.7 if x else 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}", flush=True)


def _make_per_class_curve(
    agg_perclass: dict,   # {fraction: {n_train, class_f1s: {ALDE: [...], ...}}}
    out_path: str,
) -> None:
    """F1 par classe vs proportion (Tier 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils import CLASS_NAMES

    classes = [c for c in CLASS_NAMES if c != "RHOL"]  # 11 classes
    fracs = sorted(agg_perclass.keys())
    x = [agg_perclass[f]["n_train"] for f in fracs]

    n_cls = len(classes)
    ncols = 4
    nrows = (n_cls + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)
    axes_flat = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes_flat[i]
        y_vals = []
        for f in fracs:
            vals = agg_perclass[f].get("class_f1s", {}).get(cls, [])
            y_vals.append(np.mean(vals) if vals else np.nan)
        ax.semilogx(x, y_vals, "o-", linewidth=1.8, markersize=5)
        ax.set_title(cls, fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(which="both", alpha=0.3)
        ax.tick_params(labelsize=8)

    # Cacher les axes vides
    for ax in axes_flat[n_cls:]:
        ax.set_visible(False)

    fig.text(0.5, 0.02, "Tuiles d'entraînement (log)", ha="center", fontsize=11)
    fig.text(0.02, 0.5, "F1 par classe (test)", va="center",
             rotation="vertical", fontsize=11)
    fig.suptitle("F1 par classe vs taille du train set — ViT-B/16 full FT",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Gate de reproductibilité (100%)
# ──────────────────────────────────────────────────────────────────────────────

def _check_gate(rows: list[dict], ref: float = 0.4796, tol: float = 0.005) -> None:
    runs_100 = [r for r in rows if abs(r.get("fraction", 0) - 1.0) < 1e-6]
    if not runs_100:
        print("\n[Gate] SKIP — aucun run à 100% trouvé.", flush=True)
        return
    f1_values = [r["f1_macro_pres_test"] for r in runs_100]
    f1_mean = np.mean(f1_values)
    ecart = abs(f1_mean - ref)
    status = "PASS" if ecart <= tol else "FAIL"
    print(f"\n[Gate reproductibilité 100%]")
    print(f"  vitb16_fulft_arctic ref = {ref:.4f}")
    print(f"  run(s) 100%   mean f1_pres = {f1_mean:.4f}  (valeurs: {[round(v,4) for v in f1_values]})")
    print(f"  Écart = {ecart:.4f}  tolérance = {tol:.4f}  → {status}")
    if status == "FAIL":
        print("  [ATTENTION] Divergence de recette détectée — vérifier les hyperparamètres!", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="outputs/datacurve/runs",
                    help="répertoire contenant les sous-dossiers frac*_seed*/metrics.json")
    ap.add_argument("--emb-dir", default="embeddings",
                    help="répertoire des embeddings locaux (baselines gelées)")
    ap.add_argument("--out-dir", default="outputs/datacurve",
                    help="répertoire de sortie des CSV et figures")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="ne recompute pas les baselines (si déjà dans baselines.csv)")
    ap.add_argument("--no-perclass", action="store_true",
                    help="saute la figure per_class_curve (T2)")
    args = ap.parse_args()

    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, code_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # ─── 1. Lire tous les runs ────────────────────────────────────────────────
    print("\n── Lecture des runs ──────────────────────────────────────────────")
    rows = _read_runs(args.runs_dir)
    if not rows:
        print("[WARN] Aucun run trouvé dans", args.runs_dir, flush=True)
        print("  → Exécutez d'abord les jobs SLURM sur Narval.", flush=True)

    # Gate 100%
    _check_gate(rows)

    # ─── 2. Baselines gelées ─────────────────────────────────────────────────
    print("\n── Baselines gelées ──────────────────────────────────────────────")
    baselines_path = os.path.join(args.out_dir, "baselines.csv")
    baselines: dict = {}

    if args.skip_baselines and os.path.exists(baselines_path):
        with open(baselines_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                baselines[row["model"]] = {
                    "f1_macro_pres_test": float(row["f1_macro_pres_test"]),
                    "f1_macro_8cls_test": float(row["f1_macro_8cls_test"]),
                }
        print(f"  Baselines chargées depuis {baselines_path}", flush=True)
    else:
        emb_dir = args.emb_dir
        for model_key in BASELINE_MODELS:
            tr_path = os.path.join(emb_dir, f"{model_key}_train.npy")
            if not os.path.exists(tr_path):
                print(f"  [SKIP] {model_key}: embeddings absents ({tr_path})", flush=True)
                continue
            baselines[model_key] = _probe_baseline(emb_dir, model_key)

        if baselines:
            with open(baselines_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "f1_macro_pres_test",
                                                        "f1_macro_8cls_test", "best_C"])
                writer.writeheader()
                for key in BASELINE_MODELS:
                    if key in baselines:
                        writer.writerow(baselines[key])
            print(f"  → {baselines_path}", flush=True)
        else:
            print("  [WARN] Aucune baseline calculée (embeddings manquants?)", flush=True)

    # ─── 3. CSV brut ─────────────────────────────────────────────────────────
    print("\n── CSV résultats ─────────────────────────────────────────────────")
    raw_path = os.path.join(args.out_dir, "results_raw.csv")
    fieldnames_raw = ["proportion", "n_train_tiles", "seed", "best_C", "best_epoch",
                      "f1_macro_pres_val", "f1_macro_pres_test",
                      "f1_macro_8cls_test", "accuracy_test"]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_raw)
        writer.writeheader()
        for r in sorted(rows, key=lambda x: (x.get("fraction", 0), x.get("seed", 0))):
            writer.writerow({
                "proportion":         r.get("fraction", ""),
                "n_train_tiles":      r.get("n_train_tiles", ""),
                "seed":               r.get("seed", ""),
                "best_C":             r.get("best_C", ""),
                "best_epoch":         r.get("best_epoch", ""),
                "f1_macro_pres_val":  r.get("f1_macro_pres_val", ""),
                "f1_macro_pres_test": r.get("f1_macro_pres_test", ""),
                "f1_macro_8cls_test": r.get("f1_macro_8cls_test", ""),
                "accuracy_test":      r.get("accuracy_test", ""),
            })
    print(f"  → {raw_path}  ({len(rows)} runs)", flush=True)

    # ─── 4. CSV agrégé ───────────────────────────────────────────────────────
    agg_path = os.path.join(args.out_dir, "results_agg.csv")
    # Grouper par proportion
    from collections import defaultdict
    by_frac: dict = defaultdict(list)
    for r in rows:
        by_frac[r.get("fraction", 0)].append(r)

    agg: dict = {}
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "proportion", "n_train_tiles",
            "f1_pres_mean", "f1_pres_std",
            "f1_8cls_mean", "f1_8cls_std",
        ])
        writer.writeheader()
        for frac in sorted(by_frac.keys()):
            grp = by_frac[frac]
            n_train = grp[0].get("n_train_tiles", 0)
            f1_pres = [g["f1_macro_pres_test"] for g in grp
                       if "f1_macro_pres_test" in g]
            f1_8cls = [g["f1_macro_8cls_test"] for g in grp
                       if "f1_macro_8cls_test" in g]
            row = {
                "proportion":     frac,
                "n_train_tiles":  n_train,
                "f1_pres_mean":   round(np.mean(f1_pres), 6) if f1_pres else "",
                "f1_pres_std":    round(np.std(f1_pres), 6) if f1_pres else "",
                "f1_8cls_mean":   round(np.mean(f1_8cls), 6) if f1_8cls else "",
                "f1_8cls_std":    round(np.std(f1_8cls), 6) if f1_8cls else "",
            }
            writer.writerow(row)
            agg[frac] = {
                "n_train":      n_train,
                "f1_pres_mean": np.mean(f1_pres) if f1_pres else np.nan,
                "f1_pres_std":  np.std(f1_pres) if f1_pres else np.nan,
                "f1_8cls_mean": np.mean(f1_8cls) if f1_8cls else np.nan,
                "f1_8cls_std":  np.std(f1_8cls) if f1_8cls else np.nan,
            }
    print(f"  → {agg_path}", flush=True)

    # ─── 5. Figures ──────────────────────────────────────────────────────────
    if not agg:
        print("\n[SKIP figures] Aucune donnée agrégée.", flush=True)
    else:
        print("\n── Figures ───────────────────────────────────────────────────────")
        fracs_sorted = sorted(agg.keys())
        x = [agg[f]["n_train"] for f in fracs_sorted]

        # Croisements pour chaque métrique
        for metric, mean_key in (("pres", "f1_pres_mean"), ("8cls", "f1_8cls_mean")):
            y_mean = [agg[f][mean_key] for f in fracs_sorted]
            val_key = "f1_macro_pres_test" if metric == "pres" else "f1_macro_8cls_test"
            crossovers: dict = {}
            for key in BASELINE_MODELS:
                if key not in baselines:
                    crossovers[key] = None
                    continue
                ref = baselines[key][val_key]
                cross = _interpolate_crossover(x, y_mean, ref)
                crossovers[key] = cross

            print(f"\n  Croisements ({metric}):", flush=True)
            for key, cross in crossovers.items():
                label = baselines.get(key, {}).get(val_key, "?")
                print(f"    {key}: ref={label:.4f}  "
                      + (f"croisement à {cross:,} tuiles" if cross else "jamais atteint"),
                      flush=True)

            out_fig = os.path.join(args.out_dir, f"learning_curve_{metric}.png")
            _make_learning_curve(agg, baselines, metric, out_fig, crossovers)

        # ── Per-class curve (Tier 2) ──────────────────────────────────────────
        if not args.no_perclass:
            print("\n  Per-class curve (Tier 2)...", flush=True)
            # Lire f1_per_class_test directement depuis chaque metrics.json
            agg_perclass: dict = {}
            for frac in fracs_sorted:
                pct = int(round(frac * 100))
                class_f1s_all: dict = {}
                for seed in (0, 1, 2):
                    tag = f"frac{pct:03d}_seed{seed}"
                    m_path = os.path.join(args.runs_dir, tag, "metrics.json")
                    pc = _perclass_from_metrics(m_path)
                    for cls, val in pc.items():
                        class_f1s_all.setdefault(cls, []).append(val)
                if class_f1s_all:
                    agg_perclass[frac] = {
                        "n_train": agg[frac]["n_train"],
                        "class_f1s": class_f1s_all,
                    }

            if agg_perclass:
                out_pc = os.path.join(args.out_dir, "per_class_curve.png")
                _make_per_class_curve(agg_perclass, out_pc)
            else:
                print("  [SKIP] f1_per_class_test absent des metrics.json (Tier 2 non généré)",
                      flush=True)

    # ─── 6. Récapitulatif CHANGELOG ──────────────────────────────────────────
    n_completed = len(rows)
    n_expected = 18

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║          CHANGELOG datacurve (auto-généré)                ║
╠═══════════════════════════════════════════════════════════╣
Runs complétés             : {n_completed}/{n_expected}
Gate repro 100%            : (voir ci-dessus)
SLURM array utilisé        : oui — scripts/slurm_datacurve.sh

CHIFFRES
--------
results_raw.csv            : {os.path.join(args.out_dir, "results_raw.csv")}
results_agg.csv            : {os.path.join(args.out_dir, "results_agg.csv")}
baselines.csv              : {baselines_path}

FIGURES
-------
learning_curve_pres.png    : {os.path.join(args.out_dir, "learning_curve_pres.png")}
learning_curve_8cls.png    : {os.path.join(args.out_dir, "learning_curve_8cls.png")}
per_class_curve.png        : {os.path.join(args.out_dir, "per_class_curve.png")} (si Tier 2 OK)
""")


if __name__ == "__main__":
    main()
