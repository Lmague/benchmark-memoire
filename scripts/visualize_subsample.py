#!/usr/bin/env python3
"""Visualisation sous-échantillonnée de l'espace latent (UMAP + PCA + voisinage k-NN).

Objectif : explorer visuellement la structure de l'espace latent
SAUF classe RHOL (absente du test, seule 152 tuiles en train) avec
un sous-échantillon stratifié de N points par classe → ~11*N points
au total. Assez pour visualiser, pas trop pour ne pas saturer la figure.

Par défaut N=15 (UMAP/PCA) et N=100 (matrice de voisinage : avec 15
points/classe, k=10 n'a pas de sens — au plus 14 voisins disponibles).

Sorties (dans ``figures/subsample_<N>pC/``) :
  - ``<model>_pca.png``                 PCA 2D
  - ``<model>_umap.png``                UMAP 2D
  - ``<model>_knn_heatmat.png``         Heatmap de voisinage k-NN (P(voisin=j|req=i))
  - ``<model>_knn_heatmat.csv``         Mêmes valeurs en CSV
  - ``subsample.log``                   Log d'exécution

Idempotent : skip si la PNG existe déjà. ``--force`` pour forcer.

Exemples :
  python scripts/visualize_subsample.py
  python scripts/visualize_subsample.py --n-per-class 30 --models dinov3_vitl16_lvd
  python scripts/visualize_subsample.py --knn-n-per-class 200 --knn-k 15 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

PROJ = Path(__file__).resolve().parents[1]
EMB_DIR = PROJ / "embeddings"

# 12 classes du benchmark. RHOL = idx 7 (exclue de cette analyse).
CLASS_NAMES = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS",
               "PETF", "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
EXCLUDED = {"RHOL"}  # 152 tuiles en train, 0 en val/test
DEFAULT_MODELS = ["vitb16_fulft_arctic", "dinov3_vitl16_lvd"]
RANDOM_STATE = 42

# UMAP : valeurs typiques pour visualisation, OK sur ~150-2000 points
UMAP_KW = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=RANDOM_STATE)
# k-NN voisinage
DEFAULT_KNN_K = 10
DEFAULT_KNN_NPC = 100  # points/classe pour la heatmap


# ============================================================ helpers
def _log(msg: str, log_fh) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def _palette(n: int) -> list:
    """tab20 a 20 couleurs → on prend n teintes bien espacées."""
    return [plt.cm.tab20(i / max(n - 1, 1)) for i in range(n)]


def _stratified_subsample(y: np.ndarray, n_per_class: int, seed: int) -> np.ndarray:
    """Retourne les indices d'un sous-échantillon stratifié, classes exclues sautées."""
    rng = np.random.RandomState(seed)
    keep_classes = [i for i, c in enumerate(CLASS_NAMES) if c not in EXCLUDED]
    idx_out = []
    for ci in keep_classes:
        pool = np.where(y == ci)[0]
        take = min(n_per_class, len(pool))
        if take == 0:
            continue
        idx_out.extend(rng.choice(pool, size=take, replace=False))
    return np.array(idx_out, dtype=np.int64)


def _load_split(model_key: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Charge embeddings + labels. Float32 pour sklearn (fp16 -> float32)."""
    E = np.load(EMB_DIR / f"{model_key}_{split}.npy").astype(np.float32)
    L = np.load(EMB_DIR / f"{model_key}_{split}_labels.npy").astype(np.int64)
    return E, L


# ============================================================ figures
def fig_pca(E: np.ndarray, L: np.ndarray, classes: list[str], colors: list,
            title: str, out_png: Path, log_fh) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(E)
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=130)
    for ci, cname in enumerate(classes):
        m = (L == CLASS_NAMES.index(cname))
        ax.scatter(Z[m, 0], Z[m, 1], s=42, c=[colors[ci]],
                   alpha=0.85, edgecolors="black", linewidths=0.4,
                   label=f"{cname} (n={int(m.sum())})")
    ev1 = float(pca.explained_variance_ratio_[0])
    ev2 = float(pca.explained_variance_ratio_[1])
    ax.set_xlabel(f"PC1 ({ev1*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({ev2*100:.1f}% var.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9, markerscale=0.8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    _log(f"  PCA  -> {out_png.name}  (PC1={ev1*100:.1f}%, PC2={ev2*100:.1f}%)", log_fh)


def fig_umap(E: np.ndarray, L: np.ndarray, classes: list[str], colors: list,
             title: str, out_png: Path, log_fh) -> None:
    import umap
    t0 = time.time()
    Z = umap.UMAP(**UMAP_KW).fit_transform(E)
    dt = time.time() - t0
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=130)
    for ci, cname in enumerate(classes):
        m = (L == CLASS_NAMES.index(cname))
        ax.scatter(Z[m, 0], Z[m, 1], s=42, c=[colors[ci]],
                   alpha=0.85, edgecolors="black", linewidths=0.4,
                   label=f"{cname} (n={int(m.sum())})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9, markerscale=0.8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    _log(f"  UMAP -> {out_png.name}  ({dt:.1f}s)", log_fh)


def fig_knn_heatmat(E: np.ndarray, L: np.ndarray, classes: list[str], colors: list,
                    k: int, title: str, out_png: Path, out_csv: Path,
                    log_fh) -> np.ndarray:
    """Matrice de confusion de voisinage : M[i,j] = P(j ∈ kNN | req = i)."""
    keep_idx = [CLASS_NAMES.index(c) for c in classes]
    n_c = len(classes)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(E)
    _, idx = nn.kneighbors(E)
    # exclure le self (1er voisin)
    idx = idx[:, 1:]

    # Reindexer les labels vers [0..n_c-1] pour la matrice.
    # NB : `idx` contient des indices de POSITION dans E (sous-échantillon k-NN),
    # pas des labels — il faut indexer L_r[idx], pas remapper idx directement.
    remap = {ci: i for i, ci in enumerate(keep_idx)}
    L_r = np.array([remap[v] for v in L], dtype=np.int64)
    idx_r = L_r[idx]   # (n, k) labels re-mappés

    M = np.zeros((n_c, n_c), dtype=np.float64)
    for i in range(n_c):
        m = (L_r == i)
        if not m.any():
            continue
        # pour chaque tuile de classe i, distribution de ses k voisins
        M[i] = np.bincount(idx_r[m].ravel(), minlength=n_c) / (m.sum() * k)

    # figure : heatmap avec couleurs de classe sur les axes
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=130)
    im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0,
                   aspect="auto")
    # coloriser ticks = classes
    for tick_i, (cname, col) in enumerate(zip(classes, colors)):
        ax.get_xticklabels()  # force init
        ax.get_yticklabels()
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels(classes, fontsize=9)
    # barres colorées à côté des ticks
    for spine, side in [("left", "y"), ("bottom", "x")]:
        pass  # pas de barres, on met des micro-rectangles via collections
    # rectangles de couleur (hachures) — sobre : juste un patch par tick
    for i, col in enumerate(colors):
        ax.add_patch(plt.Rectangle((i - 0.5, -1.1), 1, 0.18,
                                    color=col, clip_on=False, zorder=5))
        ax.add_patch(plt.Rectangle((-1.1, i - 0.5), 0.18, 1,
                                    color=col, clip_on=False, zorder=5))
    # annotations % (avec seuil lisible)
    thresh = M.max() * 0.5
    for i in range(n_c):
        for j in range(n_c):
            v = M[i, j]
            txt = f"{v*100:.0f}" if v >= 0.005 else ""
            color = "white" if v > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8)
    # pureté diagonale en commentaire — sur deux lignes pour éviter la troncature
    diag = np.diag(M)
    purity_str1 = " | ".join(f"{c}={diag[i]*100:.0f}%" for i, c in enumerate(classes[:6]))
    purity_str2 = " | ".join(f"{c}={diag[i]*100:.0f}%" for i, c in enumerate(classes[6:]))
    ax.set_xlabel(f"Voisin (classe j) — pureté diag. :\n{purity_str1}\n{purity_str2}",
                  fontsize=8)
    ax.set_ylabel("Requête (classe i)")
    ax.set_title(f"{title}\nk-NN cosine k={k}, matrice normalisée par requête",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(j ∈ kNN | i)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    _log(f"  kNN -> {out_png.name}  (k={k})", log_fh)

    # CSV
    with open(out_csv, "w") as f:
        f.write("query_class," + ",".join(classes) + "\n")
        for i, c in enumerate(classes):
            f.write(c + "," + ",".join(f"{M[i, j]:.6f}" for j in range(n_c)) + "\n")
    _log(f"  CSV  -> {out_csv.name}", log_fh)
    return M


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"Modèles à visualiser (défaut: {DEFAULT_MODELS})")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Split source (défaut: train — seul split avec 11 classes)")
    ap.add_argument("--n-per-class", type=int, default=15,
                    help="Points par classe pour UMAP/PCA (défaut: 15)")
    ap.add_argument("--knn-n-per-class", type=int, default=DEFAULT_KNN_NPC,
                    help=f"Points par classe pour la heatmap k-NN (défaut: {DEFAULT_KNN_NPC})")
    ap.add_argument("--knn-k", type=int, default=DEFAULT_KNN_K,
                    help=f"k du k-NN (défaut: {DEFAULT_KNN_K})")
    ap.add_argument("--out-dir", default=None,
                    help="Dossier de sortie (défaut: figures/subsample_<N>pC)")
    ap.add_argument("--force", action="store_true",
                    help="Régénère même si les PNG existent déjà")
    args = ap.parse_args()

    classes = [c for c in CLASS_NAMES if c not in EXCLUDED]  # 11 classes
    colors = _palette(len(classes))
    out_dir = (PROJ / "figures" / f"subsample_{args.n_per_class}pC" if args.out_dir is None
               else Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "subsample.log"
    if args.force and log_path.exists():
        log_path.unlink()
    log_fh = open(log_path, "a")

    _log(f"=== visualize_subsample.py — démarrage ===", log_fh)
    _log(f"  models         = {args.models}", log_fh)
    _log(f"  split          = {args.split}", log_fh)
    _log(f"  n_per_class    = {args.n_per_class}  (-> 11 × {args.n_per_class} = {11*args.n_per_class} pts)", log_fh)
    _log(f"  knn_n_per_class= {args.knn_n_per_class}", log_fh)
    _log(f"  knn_k          = {args.knn_k}", log_fh)
    _log(f"  out_dir        = {out_dir}", log_fh)
    _log(f"  classes        = {classes}", log_fh)
    _log(f"  exclues        = {sorted(EXCLUDED)}", log_fh)

    rng = np.random.RandomState(RANDOM_STATE)

    for model_key in args.models:
        _log(f"\n--- {model_key} ---", log_fh)
        try:
            E, L = _load_split(model_key, args.split)
        except Exception as exc:
            _log(f"  ! chargement KO : {exc}", log_fh)
            continue
        _log(f"  chargé : E={E.shape}  L={L.shape}  "
             f"classes présentes={len(np.unique(L))}", log_fh)

        # Sous-échantillon UMAP/PCA
        idx_umap = _stratified_subsample(L, args.n_per_class, RANDOM_STATE)
        Eu, Lu = E[idx_umap], L[idx_umap]
        _log(f"  sous-éch. UMAP/PCA : {len(idx_umap)} points", log_fh)

        # ----- PCA
        pca_png = out_dir / f"{model_key}_pca.png"
        if pca_png.exists() and not args.force:
            _log(f"  PCA  existe, skip (--force pour régénérer)", log_fh)
        else:
            fig_pca(Eu, Lu, classes, colors,
                    f"PCA — {model_key} ({args.split}, {len(idx_umap)} pts, 11 classes)",
                    pca_png, log_fh)

        # ----- UMAP
        umap_png = out_dir / f"{model_key}_umap.png"
        if umap_png.exists() and not args.force:
            _log(f"  UMAP existe, skip (--force pour régénérer)", log_fh)
        else:
            fig_umap(Eu, Lu, classes, colors,
                     f"UMAP — {model_key} ({args.split}, {len(idx_umap)} pts, 11 classes)",
                     umap_png, log_fh)

        # ----- k-NN heatmap (sous-échantillon plus large)
        idx_knn = _stratified_subsample(L, args.knn_n_per_class, RANDOM_STATE)
        Ek, Lk = E[idx_knn], L[idx_knn]
        _log(f"  sous-éch. k-NN    : {len(idx_knn)} points", log_fh)
        knn_png = out_dir / f"{model_key}_knn_heatmat.png"
        knn_csv = out_dir / f"{model_key}_knn_heatmat.csv"
        if knn_png.exists() and not args.force:
            _log(f"  kNN  existe, skip (--force pour régénérer)", log_fh)
        else:
            fig_knn_heatmat(Ek, Lk, classes, colors, args.knn_k,
                            f"k-NN voisinage — {model_key} ({args.split})",
                            knn_png, knn_csv, log_fh)

    log_fh.close()
    print(f"\nTerminé. Sorties dans : {out_dir}")


if __name__ == "__main__":
    main()
