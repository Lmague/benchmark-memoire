"""Génération des figures (PNG) à partir des JSON de résultats + features cachées.

Backend matplotlib non interactif (Agg). seaborn/pandas NE SONT PAS requis (matplotlib pur),
afin de générer les figures sur une machine d'analyse légère. Chaque fonction prend des
données déjà chargées et écrit un PNG ; l'orchestration vit dans ``make_figures.py``.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import CLASS_NAMES, ensure_dir  # noqa: E402


def _save(fig, out_path: str, dpi: int = 120) -> str:
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm, class_names: list[str], out_path: str, title: str = "") -> str:
    """Heatmap de la matrice de confusion (matplotlib pur, annotations entières)."""
    cm = np.asarray(cm)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(0.7 * n + 2, 0.7 * n + 1))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title(title or "Matrice de confusion")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


def plot_training_curves(history: dict, out_path: str, title: str = "") -> str:
    """Courbes loss (train) et F1-Macro / F1-Weighted / accuracy (val) par epoch."""
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], "-o", ms=3, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss d'entraînement")
    ax1.legend()
    for k, lbl in [("val_f1_macro", "val F1-Macro"), ("val_f1_weighted", "val F1-Weighted"),
                   ("val_acc", "val accuracy")]:
        if history.get(k):
            ax2.plot(epochs, history[k], "-o", ms=3, label=lbl)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_title("Métriques de validation")
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_f1_barplot(model_to_f1: dict, out_path: str, title: str = "F1-Macro par modèle",
                    ylabel: str = "F1-Macro") -> str:
    """Barplot horizontal d'un score (ex. F1) par modèle, trié décroissant."""
    items = sorted(model_to_f1.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    ax.barh(range(len(names)), vals, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel); ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_anisotropy_vs_f1(points: list[tuple[str, float, float]], out_path: str,
                          title: str = "Anisotropie vs F1") -> str:
    """Scatter (anisotropie, F1) annoté par modèle."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, aniso, f1 in points:
        ax.scatter(aniso, f1, s=60)
        ax.annotate(name, (aniso, f1), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Anisotropie (cosinus moyen)")
    ax.set_ylabel("F1-Macro (probe, test)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_layerwise_curves(curves: dict, out_path: str, model: str = "") -> str:
    """Courbes RankMe et anisotropie en fonction de la profondeur (par couche)."""
    layers = sorted(int(li) for li in curves)
    rankme = [curves[str(li)]["rankme"] if str(li) in curves else curves[li]["rankme"]
              for li in layers]
    aniso = [curves[str(li)]["anisotropy"] if str(li) in curves else curves[li]["anisotropy"]
             for li in layers]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(layers, rankme, "-o", ms=4, color="darkorange")
    ax1.set_xlabel("bloc transformer"); ax1.set_ylabel("RankMe"); ax1.set_title("RankMe par couche")
    ax2.plot(layers, aniso, "-o", ms=4, color="teal")
    ax2.set_xlabel("bloc transformer"); ax2.set_ylabel("anisotropie")
    ax2.set_title("Anisotropie par couche")
    fig.suptitle(f"Couche-par-couche — {model}")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_tsne(feats_per_model: dict, out_path: str, class_names: list[str] = CLASS_NAMES,
              n_per_class: int = 1000 // 12, seed: int = 42, subtitles: dict | None = None) -> str:
    """Grille de scatters t-SNE (test) par modèle, sous-échantillon stratifié (~1000 pts)."""
    from sklearn.manifold import TSNE
    keys = list(feats_per_model)
    ncols = 3
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    rng = np.random.RandomState(seed)
    for ax, key in zip(axes, keys):
        E, y = feats_per_model[key]
        idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            idx.extend(rng.choice(ci, min(n_per_class, len(ci)), replace=False))
        idx = np.asarray(idx)
        Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(E[idx])
        ys = y[idx]
        for ci, cls in enumerate(class_names):
            m = ys == ci
            if m.any():
                ax.scatter(Z[m, 0], Z[m, 1], c=[colors[ci]], s=8, alpha=0.7, label=cls)
        sub = f"\n{subtitles[key]}" if subtitles and key in subtitles else ""
        ax.set_title(f"{key}{sub}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(keys):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, out_path)
