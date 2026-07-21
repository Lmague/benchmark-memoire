#!/usr/bin/env python3
"""CKA linéaire inter-domaine, couche par couche : Arctic-test vs COCO val2017 (proxy domaine
de pré-entraînement), pour un backbone ViT FROZEN.

Objectif diagnostic (EXPLORATOIRE, design SSL Paper 2 — PAS destiné à publication en l'état) :
localiser à quelle profondeur la représentation frozen diverge le plus entre le domaine de
pré-entraînement (COCO = photos naturelles) et le domaine cible (tuiles UAV arctiques
2.24mm/px). CKA basse à la couche li = la structure de similarité intra-batch du modèle à
cette profondeur est très différente entre les deux domaines (candidat "point de rupture" à
regarder de près pour l'adaptation ExPLoRA) ; CKA haute = la représentation à cette couche
est déjà quasi-partagée entre les deux domaines.

Pourquoi ce script ne réutilise PAS ``src.features.extract_layerwise`` pour COCO :
son dataloader (``ArcticTVCDataset``, via ``_eval_loader``) exige un CSV ``(filepath,label)``
sous ``cfg.paths.csv_dir`` + tuiles sous ``cfg.paths.tiles_dir`` — câblé sur le format de split
Arctic-TVC, pas un simple dossier d'images. Forcer COCO là-dedans demanderait de fabriquer un
CSV avec des labels factices, ce qui n'est pas un usage légitime du format. Ce script fait donc
son propre forward pass minimal sur COCO, en réutilisant ce qui EST générique dans le repo :
``build_frozen_extractor`` (loader HF AutoModel), ``get_transformer_blocks`` (repérage des
blocs transformer, gère déjà la convention HF DINOv3 ``model.model.layer``) et
``build_transforms`` (Compose torchvision générique). Côté Arctic, les embeddings layerwise
sont ceux déjà cachés par ``extract_layerwise`` (lecture seule via ``load_layerwise``).

``linear_cka`` est copiée à l'identique depuis ``scripts/layerwise_probe.py`` (pas d'import
cross-script — fragile hors invocation directe en ``python scripts/...`` — ni de modification
de ``layerwise_probe.py``).

Prérequis : embeddings layerwise Arctic ``test`` déjà extraits pour le modèle visé
(``extract_layerwise`` / pipeline existant). Images COCO val2017 déjà téléchargées sur disque
(voir commandes proposées séparément — téléchargement sur login node, PAS dans ce script).

Usage :
  python scripts/cka_domain_shift.py \\
      --config configs/frozen_eval.yaml \\
      --model dinov3_vitb16_lvd \\
      --coco-dir /lustre07/scratch/lmague/coco/val2017 \\
      --subsample-n 5000 \\
      --output results/cka_domain_shift.json \\
      --fig results/figures/cka_domain_shift.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import count_layerwise_layers, load_layerwise


# --------------------------------------------------------------------------------- linear_cka
def linear_cka(X, Y):
    """CKA linéaire (Kornblith et al. 2019) entre deux matrices de représentation
    aux MÊMES lignes (échantillons), dimensions de features éventuellement différentes.

    Retourne une similarité dans [0, 1] : proche de 1 = représentations quasi identiques
    (bloc "redondant") ; proche de 0 = fort changement représentationnel.

    Formulation feature-space : HSIC_lin(X, Y) = ||X^T Y||_F^2 ; efficace quand d << n.
    Les deux matrices doivent être alignées ligne-à-ligne (mêmes tuiles dans le même ordre).

    Copie à l'identique de ``scripts/layerwise_probe.py::linear_cka`` — voir docstring module.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(((X.T @ Y) ** 2).sum())
    hsic_xx = float(((X.T @ X) ** 2).sum())
    hsic_yy = float(((Y.T @ Y) ** 2).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else float("nan")


# --------------------------------------------------------------------------------------- COCO
def _list_coco_images(coco_dir: str) -> list[str]:
    """Liste triée (déterministe) des images dans ``coco_dir`` (val2017 = .jpg à plat)."""
    paths: list[str] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(coco_dir, pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"Aucune image trouvée dans {coco_dir} (attendu des .jpg COCO val2017 à plat).")
    return paths


def _extract_coco_layerwise(cfg, model_key: str, image_paths: list[str]) -> tuple[dict, int]:
    """Forward pass minimal sur COCO : charge le modèle via le loader générique du repo
    (``build_frozen_extractor`` + ``get_transformer_blocks``), capture le CLS en sortie de
    chaque bloc transformer (même convention de hook que ``features.extract_layerwise``,
    réimplémentée ici en autonome — PAS d'import de cette fonction, elle est câblée sur
    ArcticTVCDataset). Retourne ``({layer_idx: E float32 (N,dim)}, n_layers)``.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.data import build_transforms
    from src.models import build_frozen_extractor, get_transformer_blocks
    from src.utils import get_device, get_normalization

    class _ImageFolderDataset(Dataset):
        """Dataset minimal : liste de chemins d'images, PAS de CSV/labels (contrairement à
        ArcticTVCDataset — COCO n'a pas de schéma de labels Arctic-TVC, et la CKA n'en a pas
        besoin)."""

        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(self.paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            return self.transform(img)

    model, forward_fn, dim, norm_key = build_frozen_extractor(model_key, cfg.raw.get("checkpoint"))
    device = get_device()
    model = model.to(device).eval()
    blocks = get_transformer_blocks(model)
    n_layers = len(blocks)

    expected = 24 if "vitl16" in model_key.lower() else 12 if "vitb16" in model_key.lower() else None
    if expected is not None and n_layers != expected:
        raise RuntimeError(
            f"[cka-domain-shift] {model_key}: {n_layers} blocs captés, {expected} attendus "
            "pour cette architecture — mauvaise ModuleList (structure HF inattendue ?).")

    mean, std = get_normalization(norm_key)
    tf = build_transforms("eval", mean, std, cfg.data.image_size)
    ds = _ImageFolderDataset(image_paths, tf)
    loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, pin_memory=True)

    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    for li in range(n_layers):
        def _hook(_m, _inp, out, li=li):
            o = out[0] if isinstance(out, tuple) else out
            captured[li] = o[:, 0, :].detach().cpu().to(torch.float16)  # token CLS

        handles.append(blocks[li].register_forward_hook(_hook))

    per_layer = {li: [] for li in range(n_layers)}
    try:
        with torch.no_grad():
            for x in loader:
                x = x.to(device, non_blocking=True)
                captured.clear()
                forward_fn(model, x)
                for li in range(n_layers):
                    per_layer[li].append(captured[li].numpy())
    finally:
        for h in handles:
            h.remove()

    return ({li: np.concatenate(per_layer[li], axis=0).astype(np.float32) for li in range(n_layers)},
           n_layers)


# ------------------------------------------------------------------------------------- figure
def make_domain_shift_figure(result: dict, fig_path: str) -> None:
    """CKA(Arctic-test, COCO) par couche — bas = divergence de domaine forte à cette profondeur.

    Style aligné sur ``layerwise_probe.make_depth_figure`` (mêmes conventions visuelles).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [d["layer"] for d in result["per_layer"]]
    ys = [d["cka_domain_shift"] for d in result["per_layer"]]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(xs, ys, marker="o", ms=3, linewidth=1.0, alpha=0.85, label=result["model"])
    ax.axvline(result["min_cka_layer"], color="crimson", ls="--", lw=0.8, alpha=0.6,
              label=f"min @ L{result['min_cka_layer']}")
    ax.set_xlabel("Index de couche (bloc transformer)", fontsize=10)
    ax.set_ylabel("CKA linéaire (Arctic-test vs COCO val2017)\n(bas = forte divergence de domaine)",
                  fontsize=10)
    ax.set_title("Shift de domaine par profondeur — frozen pré-entraîné vs cible arctique",
                fontsize=11, fontweight="bold")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {fig_path}")


# ---------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--model", default="dinov3_vitb16_lvd",
                    help="Modèle frozen visé (embeddings layerwise Arctic déjà extraits).")
    ap.add_argument("--arctic-split", default="test")
    ap.add_argument("--coco-dir", required=True,
                    help="Dossier plat d'images COCO val2017 (.jpg), ex. "
                         "/lustre07/scratch/lmague/coco/val2017")
    ap.add_argument("--subsample-n", type=int, default=5000,
                    help="N cible partagé Arctic/COCO pour la CKA (réduit si l'un des deux "
                         "pools est plus petit).")
    ap.add_argument("--output", default="results/cka_domain_shift.json")
    ap.add_argument("--fig", default="results/figures/cka_domain_shift.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.train.seed

    # --- Arctic : embeddings layerwise déjà cachés (lecture seule).
    n_layers_arctic = count_layerwise_layers(cfg, args.model, args.arctic_split)
    lw_arctic = load_layerwise(cfg, args.model, args.arctic_split, n_layers_arctic)
    n_arctic = int(np.asarray(lw_arctic[0][0]).shape[0])

    # --- COCO : pool d'images disponibles sur disque.
    coco_paths_all = _list_coco_images(args.coco_dir)
    n_coco_pool = len(coco_paths_all)

    # N partagé Arctic/COCO — la CKA (linear_cka) exige des matrices avec le MÊME nombre de
    # lignes des deux côtés (mais PAS de correspondance image-à-image : elle compare des
    # structures de similarité intra-batch, pas des paires). RNG seedé sur cfg.train.seed,
    # deux tirages indépendants (populations différentes) — c'est load-bearing pour la
    # validité/reproductibilité du calcul.
    n = min(args.subsample_n, n_arctic, n_coco_pool)
    if n < args.subsample_n:
        print(f"[warn] N réduit à {n} (subsample_n={args.subsample_n}, "
              f"n_arctic={n_arctic}, n_coco_pool={n_coco_pool})")

    rng_arctic = np.random.RandomState(seed)
    idx_arctic = (rng_arctic.choice(n_arctic, n, replace=False)
                 if n_arctic > n else np.arange(n_arctic))

    rng_coco = np.random.RandomState(seed)
    idx_coco = (rng_coco.choice(n_coco_pool, n, replace=False)
               if n_coco_pool > n else np.arange(n_coco_pool))
    coco_paths = [coco_paths_all[i] for i in sorted(idx_coco.tolist())]

    E_coco, n_layers_coco = _extract_coco_layerwise(cfg, args.model, coco_paths)
    if n_layers_coco != n_layers_arctic:
        raise RuntimeError(
            f"[cka-domain-shift] Nb de couches incohérent pour {args.model}: "
            f"arctic={n_layers_arctic} coco={n_layers_coco}.")

    per_layer = []
    for li in range(n_layers_arctic):
        E_a = np.asarray(lw_arctic[li][0], dtype=np.float32)[idx_arctic]
        E_c = E_coco[li]
        assert E_c.shape[0] == n, f"L{li}: COCO N={E_c.shape[0]} != {n}"
        cka = linear_cka(E_a, E_c)
        per_layer.append({"layer": li, "cka_domain_shift": cka})
        print(f"[cka-domain-shift] {args.model} L{li:02d}  CKA(Arctic,COCO)={cka:.4f}")

    min_entry = min(per_layer, key=lambda d: d["cka_domain_shift"])
    result = {
        "model": args.model,
        "n_layers": n_layers_arctic,
        "per_layer": per_layer,
        "min_cka_layer": min_entry["layer"],
        "subsample_n": n,
        "seed": seed,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {args.output}")
    print(f"\nDivergence maximale à la couche L{min_entry['layer']:02d} "
          f"(CKA={min_entry['cka_domain_shift']:.4f})")

    make_domain_shift_figure(result, args.fig)


if __name__ == "__main__":
    main()
