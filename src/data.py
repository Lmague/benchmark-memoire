"""Dataset Arctic-TVC, augmentations (version notebook, style Moummad) et DataLoaders.

Les augmentations reproduisent EXACTEMENT le pipeline des notebooks (affines + posterisation,
flips ; PAS de rotation 90° discrète — celle-ci n'existe que dans train_supervised.py et est
écartée). La normalisation d'entrée est paramétrable par modèle (ImageNet vs DINOv3-SAT).

``torch`` est importé au niveau module (data.py n'est chargé que par les CLIs, jamais par
``import src``) ; ``torchvision`` reste paresseux pour permettre d'instancier le Dataset sans
torchvision.
"""
from __future__ import annotations

import csv
import os

from torch.utils.data import DataLoader, Dataset

from .utils import get_normalization, seed_worker


class ArcticTVCDataset(Dataset):
    """Lit un CSV ``(filepath,label)`` et charge les PNG depuis ``tiles_root``."""

    def __init__(self, csv_path: str, tiles_root: str, transform=None):
        self.tiles_root = tiles_root
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)  # header
            for row in r:
                if len(row) >= 2:
                    self.samples.append((row[0], int(row[1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        fp, lb = self.samples[idx]
        fb = os.environ.get("ARCTIC_TILES_FALLBACK", "")
        path = os.path.join(self.tiles_root, fp)
        if not os.path.exists(path) and fb:
            path = os.path.join(fb, fp)
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception:
            # Fichier absent ou corrompu (ex. écriture SHM incomplète) → fallback Drive
            if not fb:
                raise
            img = Image.open(os.path.join(fb, fp))
            if img.mode != "RGB":
                img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, lb


def build_transforms(split: str, mean=None, std=None, image_size: int = 224):
    """Transforms d'entraînement (augmentées) ou d'éval (déterministes).

    ``split='train'`` → augmentations notebook ; sinon Resize+CenterCrop déterministe.
    """
    from torchvision import transforms as T
    if mean is None or std is None:
        mean, std = get_normalization("imagenet")
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.2, hue=0.1)], p=0.8),
            T.RandomPosterize(bits=5, p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ])
    return T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std)),
    ])


def make_loaders(cfg, splits=("train", "val", "test"), train_aug: bool = True) -> dict:
    """Construit les DataLoaders. ``train_aug=False`` → transforms d'éval partout (extraction).

    Pas de WeightedRandomSampler : le déséquilibre est géré par la loss pondérée
    (cf. mémoire projet). Le train est mélangé via un Generator seedé + ``seed_worker``.
    """
    import torch
    mean, std = get_normalization(cfg.model.norm)
    loaders: dict = {}
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    for s in splits:
        is_train = (s == "train" and train_aug)
        tf = build_transforms("train" if is_train else "eval", mean, std, cfg.data.image_size)
        ds = ArcticTVCDataset(os.path.join(cfg.paths.csv_dir, f"{s}.csv"),
                              cfg.paths.tiles_dir, tf)
        loaders[s] = DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=is_train,
            generator=g if is_train else None,
            worker_init_fn=seed_worker if is_train else None,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
        )
    return loaders
