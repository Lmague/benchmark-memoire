"""Constantes du dataset et utilitaires partagés (seed, IO, normalisations, Colab).

Volontairement léger : seules numpy et la stdlib sont importées au niveau module.
torch est importé dans les fonctions qui en ont besoin (pour que ``import src``
fonctionne sans torch), et ``google.colab`` uniquement dans :func:`maybe_mount_drive`.
"""
from __future__ import annotations

import csv
import json
import os
import random

import numpy as np

# --- Classes (ordre EXACT — ne JAMAIS réordonner : les labels CSV en dépendent) ---
CLASS_NAMES: list[str] = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS",
                          "PETF", "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
N_CLASSES: int = len(CLASS_NAMES)
RHOL_IDX: int = CLASS_NAMES.index("RHOL")              # 7 — absente du split test
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# --- Normalisations d'entrée par famille de modèle ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    # DINOv3 SAT-493M : normalisation satellite dédiée (repo officiel, make_transform SAT).
    "dinov3_sat": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
    # SimDINOv2 (iNat Plantae) : stats custom du transform d'éval (sslplant/simdinov2/eval/get_data.py).
    "simdino_inat": ((0.429, 0.459, 0.328), (0.221, 0.215, 0.221)),
}


def get_normalization(name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Retourne (mean, std) pour la clé de normalisation donnée."""
    if name not in NORMALIZATIONS:
        raise KeyError(f"normalisation inconnue '{name}' (dispo: {sorted(NORMALIZATIONS)})")
    return NORMALIZATIONS[name]


def set_seed(seed: int = 42) -> None:
    """Fixe random/numpy/torch/cuda et active cudnn.benchmark."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 (signature imposée par DataLoader)
    """Graine déterministe par worker DataLoader (reproductibilité du shuffle)."""
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    """Retourne le device CUDA si disponible, sinon CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_split_csv(path: str) -> tuple[list[str], np.ndarray]:
    """Lit un CSV (filepath,label) et retourne (filepaths, labels int64)."""
    fps: list[str] = []
    lbs: list[int] = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) >= 2:
                fps.append(row[0])
                lbs.append(int(row[1]))
    return fps, np.asarray(lbs, dtype=np.int64)


def ensure_dir(path: str) -> None:
    """Crée le dossier (et parents) si non vide et inexistant."""
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    """Sérialise ``obj`` en JSON indenté (crée le dossier parent)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    """Charge un fichier JSON."""
    with open(path) as f:
        return json.load(f)


def run_tag(model: str, regime: str) -> str:
    """Identifiant de run pour nommer checkpoints/résultats (évite l'écrasement)."""
    return f"{model}_{regime}"


def rhol_passes() -> list[tuple[str, int | None, list[str]]]:
    """Les deux passes d'évaluation : (tag, label_à_retirer, noms_de_classes).

    - ``with_rhol``    : 12 classes, RHOL comptée 0 (convention).
    - ``without_rhol`` : 11 classes, RHOL (idx 7) retirée et indices comblés.
    """
    return [
        ("with_rhol", None, list(CLASS_NAMES)),
        ("without_rhol", RHOL_IDX, [c for c in CLASS_NAMES if c != "RHOL"]),
    ]


def maybe_mount_drive(env: str) -> None:
    """Monte Google Drive sur Colab uniquement.

    ``google.colab`` est importé ICI, jamais au niveau module, pour préserver
    l'import local du package (``python -c "import src"``).
    """
    if env != "colab":
        return
    from google.colab import drive  # import local volontaire (Colab seulement)
    drive.mount("/content/drive")
