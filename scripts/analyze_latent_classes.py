#!/usr/bin/env python3
"""Diagnostic latente des classes mal apprises (Q3) — UMAP + PCA + k-NN purity.

Objectif : produire des visualisations pour répondre visuellement à
- Ces classes occupent-elles des régions MIXTES (confusion inter-classe) ?
- Sont-elles COMPACTES mais mal séparées de classes voisines ?
- Sont-elles DISPERSÉES (bruit intra-classe) ?

Cibles : ARCA, DRYI, RHOL, RUBC (4 classes structurellement à ~0 F1).
Sources : ``embeddings/{model}_test.npy`` + ``embeddings/{model}_test_labels.npy``.

Sorties (lecture seule sur embeddings ; écriture uniquement dans ``figures/``) :
  - ``figures/pca_class_diag/{model}_pca_classes.png``
  - ``figures/umap_class_diag/{model}_umap_classes.png``
  - ``figures/pca_class_diag/knn_purity_summary.csv`` (append)
  - ``figures/umap_class_diag/umap_grid_target_classes.png`` (grille 4×3 si ≥ 4 modèles)

Idempotent : skip si la PNG existe déjà. Tolère les embeddings corrompus
(saut du modèle + log). UMAP peut être long : on garde une trace de timing.
"""
from __future__ import annotations

import os
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ============================================================ config
PROJ = Path(__file__).resolve().parents[1]
EMB_DIR = PROJ / "embeddings"
FIG_PCA = PROJ / "figures" / "pca_class_diag"
FIG_UMAP = PROJ / "figures" / "umap_class_diag"
CSV_KNN = FIG_PCA / "knn_purity_summary.csv"
LOG_FILE = FIG_PCA / "analyze_latent_classes.log"

CLASS_NAMES = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS",
               "PETF", "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
# 4 classes cibles (structurellement ~0 F1)
TARGET_CLASSES = ["ARCA", "DRYI", "RHOL", "RUBC"]
TARGET_IDX = [CLASS_NAMES.index(c) for c in TARGET_CLASSES]
N_CLASSES = len(CLASS_NAMES)

# Palette : cibles saturées, autres classes en gris semi-transparent
TARGET_COLORS = {
    "ARCA": "#d62728",  # rouge
    "DRYI": "#ff7f0e",  # orange
    "RHOL": "#9467bd",  # violet
    "RUBC": "#17becf",  # cyan
}
TARGET_MARKERS = {
    "ARCA": "X",
    "DRYI": "P",
    "RHOL": "D",  # pas dans test mais on garde pour la légende
    "RUBC": "*",
}
OTHER_COLOR = "#bdbdbd"
OTHER_MARKER = "."
OTHER_ALPHA = 0.18
TARGET_ALPHA = 0.85
TARGET_SIZE = 110
OTHER_SIZE = 12

RANDOM_STATE = 42
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
KNN_K = 20


# ============================================================ helpers
def _log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _discover_models() -> list[str]:
    """Liste les modèles dont on a test embeddings (.npy + labels .npy)."""
    models = []
    for p in sorted(EMB_DIR.glob("*_test.npy")):
        if p.name.startswith("."):
            continue
        key = p.name[: -len("_test.npy")]
        if (EMB_DIR / f"{key}_test_labels.npy").exists():
            models.append(key)
    return models


def _load(model_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Charge embeddings + labels test. Retourne None si KO."""
    try:
        E = np.load(EMB_DIR / f"{model_key}_test.npy").astype(np.float32)
        L = np.load(EMB_DIR / f"{model_key}_test_labels.npy").astype(np.int64)
    except Exception as exc:
        _log(f"  ! {model_key} : embedding illisible ({exc})")
        return None
    if E.ndim != 2 or E.shape[0] != L.shape[0]:
        _log(f"  ! {model_key} : shape mismatch E={E.shape} vs L={L.shape}")
        return None
    if E.shape[0] != 17598:
        _log(f"  ! {model_key} : n_test={E.shape[0]} ≠ 17598 (split ≠ v3)")
        return None
    return E, L


def _scatter_pca(ax, E, L, title: str) -> None:
    """Scatter PCA 2D : cibles saturées, autres gris."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(E)
    other = ~np.isin(L, TARGET_IDX)
    ax.scatter(Z[other, 0], Z[other, 1],
               s=OTHER_SIZE, c=OTHER_COLOR, marker=OTHER_MARKER,
               alpha=OTHER_ALPHA, edgecolors="none", rasterized=True)
    for c in TARGET_CLASSES:
        ci = CLASS_NAMES.index(c)
        m = (L == ci)
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1],
                   s=TARGET_SIZE, c=TARGET_COLORS[c],
                   marker=TARGET_MARKERS[c], alpha=TARGET_ALPHA,
                   edgecolors="black", linewidths=0.5,
                   label=f"{c} (n={int(m.sum())})", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=0.6)


def _scatter_umap(ax, E, L, title: str) -> None:
    """Scatter UMAP 2D : cibles saturées, autres gris."""
    import umap
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric="cosine", random_state=RANDOM_STATE,
                        verbose=False)
    Z = reducer.fit_transform(E)
    other = ~np.isin(L, TARGET_IDX)
    ax.scatter(Z[other, 0], Z[other, 1],
               s=OTHER_SIZE, c=OTHER_COLOR, marker=OTHER_MARKER,
               alpha=OTHER_ALPHA, edgecolors="none", rasterized=True)
    for c in TARGET_CLASSES:
        ci = CLASS_NAMES.index(c)
        m = (L == ci)
        if not m.any():
            continue
        ax.scatter(Z[m, 0], Z[m, 1],
                   s=TARGET_SIZE, c=TARGET_COLORS[c],
                   marker=TARGET_MARKERS[c], alpha=TARGET_ALPHA,
                   edgecolors="black", linewidths=0.5,
                   label=f"{c} (n={int(m.sum())})", rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=0.6)


def _knn_purity_per_class(E, L, k: int = KNN_K) -> dict[str, float]:
    """Pour chaque classe : fraction de ses k plus-proches voisins qui sont de la même classe.

    Métrique cosine (embeddings normalisés ou non — la métrique cosine est invariante
    à la norme, donc on prend les vecteurs bruts fp32).
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(E)
    _, idx = nn.kneighbors(E)
    # exclure le self (premier voisin)
    idx = idx[:, 1:]
    purity = {}
    for ci in range(N_CLASSES):
        m = (L == ci)
        if not m.any():
            purity[CLASS_NAMES[ci]] = float("nan")
            continue
        # pour chaque tuile de la classe, regarder la fraction de voisins de la même classe
        labels_neighbors = L[idx[m]]            # (n_classe, k)
        same = (labels_neighbors == ci).mean(axis=1)  # par tuile
        purity[CLASS_NAMES[ci]] = float(same.mean())
    return purity


def _process_model(model_key: str) -> tuple[dict, dict]:
    """Traite 1 modèle : PCA, UMAP, k-NN purity. Retourne (timings, purity).

    Idempotent : skip si la PNG PCA existe déjà.
    """
    out: dict = {"pca": None, "umap": None, "knn": None, "errors": []}
    pca_png = FIG_PCA / f"{model_key}_pca_classes.png"
    umap_png = FIG_UMAP / f"{model_key}_umap_classes.png"

    data = _load(model_key)
    if data is None:
        out["errors"].append("load_embedding")
        return out, {}
    E, L = data

    # ---------- PCA ----------
    t0 = time.time()
    if pca_png.exists():
        _log(f"  PCA {model_key} : {pca_png.name} existe déjà, skip")
        out["pca"] = 0.0
    else:
        try:
            fig, ax = plt.subplots(figsize=(8, 6.5), dpi=130)
            _scatter_pca(ax, E, L, title=f"PCA — {model_key} (test, 17 598 tuiles)")
            fig.tight_layout()
            fig.savefig(pca_png, bbox_inches="tight")
            plt.close(fig)
            out["pca"] = round(time.time() - t0, 1)
            _log(f"  PCA {model_key} : OK ({out['pca']}s) -> {pca_png.name}")
        except Exception as exc:
            out["errors"].append(f"pca:{exc}")
            _log(f"  PCA {model_key} : ERREUR {exc}")

    # ---------- UMAP ----------
    t0 = time.time()
    if umap_png.exists():
        _log(f"  UMAP {model_key} : {umap_png.name} existe déjà, skip")
        out["umap"] = 0.0
    else:
        try:
            fig, ax = plt.subplots(figsize=(8, 6.5), dpi=130)
            _scatter_umap(ax, E, L, title=f"UMAP — {model_key} (test)")
            fig.tight_layout()
            fig.savefig(umap_png, bbox_inches="tight")
            plt.close(fig)
            out["umap"] = round(time.time() - t0, 1)
            _log(f"  UMAP {model_key} : OK ({out['umap']}s) -> {umap_png.name}")
        except Exception as exc:
            out["errors"].append(f"umap:{exc}")
            _log(f"  UMAP {model_key} : ERREUR (fallback PCA seul) {exc}")

    # ---------- k-NN purity ----------
    t0 = time.time()
    try:
        purity = _knn_purity_per_class(E, L, k=KNN_K)
        out["knn"] = round(time.time() - t0, 1)
        # append CSV (model, class, knn_purity, n_test)
        rows = []
        counts = {ci: int((L == ci).sum()) for ci in range(N_CLASSES)}
        for cname, p in purity.items():
            ci = CLASS_NAMES.index(cname)
            rows.append({
                "model": model_key,
                "class": cname,
                "is_target": cname in TARGET_CLASSES,
                "n_test": counts[ci],
                "knn_k": KNN_K,
                "knn_purity": p,
            })
        df_new = pd.DataFrame(rows)
        if CSV_KNN.exists():
            df_old = pd.read_csv(CSV_KNN)
            # déduplique sur (model, class) — garde la dernière mesure
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["model", "class"], keep="last")
        else:
            df_all = df_new
        df_all.to_csv(CSV_KNN, index=False)
        _log(f"  k-NN purity {model_key} : OK ({out['knn']}s, {len(rows)} classes) -> {CSV_KNN.name}")
        return out, purity
    except Exception as exc:
        out["errors"].append(f"knn:{exc}")
        _log(f"  k-NN purity {model_key} : ERREUR {exc}")
        return out, {}


def _make_grid_umap(model_purities: dict[str, dict]) -> None:
    """Grille UMAP — un subplot par modèle, palette des 4 cibles seulement.

    model_purities : {model: {class: purity}}
    """
    out_png = FIG_UMAP / "umap_grid_target_classes.png"
    if out_png.exists():
        _log(f"  Grille UMAP existe déjà ({out_png.name}), skip")
        return

    model_keys = list(model_purities.keys())
    if len(model_keys) < 2:
        _log(f"  Grille UMAP sautée : {len(model_keys)} modèle(s) < 2 minimum")
        return
    # Layout : 3 colonnes × ceil(N/3) lignes
    import math
    ncol = 3
    nrow = math.ceil(len(model_keys) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow), dpi=110)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    import umap
    for ax, model_key in zip(axes_flat, model_keys):
        data = _load(model_key)
        if data is None:
            ax.set_title(f"{model_key} (load KO)")
            ax.axis("off")
            continue
        E, L = data
        try:
            Z = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                          metric="cosine", random_state=RANDOM_STATE,
                          verbose=False).fit_transform(E)
        except Exception as exc:
            ax.set_title(f"{model_key} (UMAP KO: {exc})")
            ax.axis("off")
            continue
        other = ~np.isin(L, TARGET_IDX)
        ax.scatter(Z[other, 0], Z[other, 1],
                   s=8, c=OTHER_COLOR, marker=OTHER_MARKER,
                   alpha=OTHER_ALPHA, edgecolors="none", rasterized=True)
        # cible (avec pureté en sous-titre)
        for c in TARGET_CLASSES:
            ci = CLASS_NAMES.index(c)
            m = (L == ci)
            if not m.any():
                continue
            p = model_purities[model_key].get(c, float("nan"))
            label = f"{c} (p={p:.2f})" if not np.isnan(p) else c
            ax.scatter(Z[m, 0], Z[m, 1],
                       s=TARGET_SIZE, c=TARGET_COLORS[c],
                       marker=TARGET_MARKERS[c], alpha=TARGET_ALPHA,
                       edgecolors="black", linewidths=0.5,
                       label=label, rasterized=True)
        ax.set_title(model_key, fontsize=10)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=0.5)

    # axes vides (si len non multiple de ncol)
    for ax in axes_flat[len(model_keys):]:
        ax.axis("off")

    fig.suptitle("UMAP — 4 classes cibles à travers les modèles\n"
                 "(label : pureté k-NN cosine k=20, voisinage = classe elle-même)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    _log(f"  Grille UMAP : OK -> {out_png.name}")


def _make_pair_finetuned() -> None:
    """PCA + UMAP frozen vs fine-tuned côte à côte pour vitb16_imagenet vs vitb16_fulft_arctic."""
    pairs = [("vitb16_imagenet", "vitb16_fulft_arctic")]
    for frozen_key, ft_key in pairs:
        data_f = _load(frozen_key)
        data_t = _load(ft_key)
        if data_f is None or data_t is None:
            _log(f"  Pair FT skip : {frozen_key}={data_f is not None} / {ft_key}={data_t is not None}")
            return
        E_f, L_f = data_f
        E_t, L_t = data_t
        out_png = FIG_PCA / f"pair_{frozen_key}_vs_{ft_key}_pca.png"
        out_umap = FIG_UMAP / f"pair_{frozen_key}_vs_{ft_key}_umap.png"
        if out_png.exists() and out_umap.exists():
            _log(f"  Pair FT existe déjà, skip")
            return
        # PCA
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
        _scatter_pca(axes[0], E_f, L_f, title=f"{frozen_key} — frozen (PCA)")
        _scatter_pca(axes[1], E_t, L_t, title=f"{ft_key} — fine-tuned (PCA)")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        # UMAP
        import umap
        Z_f = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric="cosine", random_state=RANDOM_STATE,
                        verbose=False).fit_transform(E_f)
        Z_t = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        metric="cosine", random_state=RANDOM_STATE,
                        verbose=False).fit_transform(E_t)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=120)
        for ax, Z, L, key in zip(axes, [Z_f, Z_t], [L_f, L_t],
                                 [frozen_key, ft_key]):
            other = ~np.isin(L, TARGET_IDX)
            ax.scatter(Z[other, 0], Z[other, 1],
                       s=OTHER_SIZE, c=OTHER_COLOR, marker=OTHER_MARKER,
                       alpha=OTHER_ALPHA, edgecolors="none", rasterized=True)
            for c in TARGET_CLASSES:
                ci = CLASS_NAMES.index(c)
                m = (L == ci)
                if not m.any():
                    continue
                ax.scatter(Z[m, 0], Z[m, 1],
                           s=TARGET_SIZE, c=TARGET_COLORS[c],
                           marker=TARGET_MARKERS[c], alpha=TARGET_ALPHA,
                           edgecolors="black", linewidths=0.5,
                           label=f"{c} (n={int(m.sum())})", rasterized=True)
            ax.set_title(f"{key} — UMAP")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=0.5)
        fig.tight_layout()
        fig.savefig(out_umap, bbox_inches="tight")
        plt.close(fig)
        _log(f"  Pair FT {frozen_key} vs {ft_key} : OK -> {out_png.name} + {out_umap.name}")


def main() -> None:
    FIG_PCA.mkdir(parents=True, exist_ok=True)
    FIG_UMAP.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    _log(f"=== analyze_latent_classes.py — démarrage ===")
    _log(f"  embeddings_dir = {EMB_DIR}")
    _log(f"  pca_out        = {FIG_PCA}")
    _log(f"  umap_out       = {FIG_UMAP}")
    _log(f"  classes cibles = {TARGET_CLASSES} (idx={TARGET_IDX})")

    models = _discover_models()
    _log(f"  Modèles découverts ({len(models)}) : {models}")

    # anomalies
    _log(f"  Test set : {17598} tuiles, RHOL (idx 7) absent (0 tuile) → info-positionnel uniquement")
    _log(f"  Effectifs cibles test : " + ", ".join(
        f"{c}=? (relevé par modèle)" for c in TARGET_CLASSES))

    # ===== Tier 1 : PCA + UMAP + k-NN purity par modèle =====
    all_purity: dict[str, dict] = {}
    failures: list[str] = []
    for mk in models:
        _log(f"\n--- {mk} ---")
        timings, purity = _process_model(mk)
        if timings.get("errors"):
            failures.append(f"{mk}: {timings['errors']}")
        if purity:
            all_purity[mk] = purity

    # ===== Tier 2.1 : grille UMAP (si ≥ 4 modèles) =====
    grid_done = False
    if len(all_purity) >= 4:
        _log(f"\n--- Grille UMAP ({len(all_purity)} modèles) ---")
        try:
            _make_grid_umap(all_purity)
            grid_done = True
        except Exception as exc:
            _log(f"  Grille UMAP : ERREUR {exc}")
            failures.append(f"umap_grid:{exc}")
    else:
        _log(f"\nGrille UMAP sautée : {len(all_purity)} modèle(s) avec k-NN OK < 4 minimum")

    # ===== Tier 2.2 : pair frozen vs FT =====
    ft_done = False
    if ("vitb16_imagenet" in all_purity) and ("vitb16_fulft_arctic" in all_purity):
        _log(f"\n--- Pair frozen vs fine-tuned ---")
        try:
            _make_pair_finetuned()
            ft_done = True
        except Exception as exc:
            _log(f"  Pair FT : ERREUR {exc}")
            failures.append(f"ft_pair:{exc}")
    else:
        _log(f"\nPair FT sauté : vitb16_imagenet={('vitb16_imagenet' in all_purity)}, "
             f"vitb16_fulft_arctic={('vitb16_fulft_arctic' in all_purity)}")

    _log(f"\n=== Terminé — {len(all_purity)} modèles traités ===")
    if failures:
        _log(f"  Échecs : {failures}")


if __name__ == "__main__":
    main()