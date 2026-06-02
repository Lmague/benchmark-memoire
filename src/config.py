"""Chargement et fusion des configurations YAML.

Une config effective = ``base.yaml`` (défauts communs) fusionné en profondeur avec
un YAML spécifique (modèle × régime, ou ``frozen_*``). Le champ ``env`` sélectionne
le bloc de chemins (``paths_local`` ou ``paths_colab``). Aucune logique métier ici :
uniquement parsing, validation des clés et résolution des chemins.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any

import yaml

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
BASE_CONFIG = os.path.join(CONFIGS_DIR, "base.yaml")


@dataclass
class Paths:
    """Chemins résolus selon l'environnement (local / Colab)."""
    data_root: str = ""
    tiles_dir: str = ""
    csv_dir: str = ""          # split v3 canonique (train/val/test.csv)
    old_csv_dir: str = ""      # ancien split : ordre des embeddings cachés
    emb_dir: str = ""          # cache features fp16 .npy
    ckpt_dir: str = ""
    results_dir: str = ""


@dataclass
class DataCfg:
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 224


@dataclass
class ModelCfg:
    name: str = "vitb16"
    num_classes: int = 12
    norm: str = "imagenet"     # clé dans utils.NORMALIZATIONS


@dataclass
class OptimCfg:
    type: str = "adamw"
    weight_decay: float = 0.05
    # groupes de params -> LR ; les clés doivent matcher celles produites par models.build_model
    lr: dict[str, float] = field(default_factory=lambda: {"backbone": 1e-5, "head": 1e-4})


@dataclass
class ScheduleCfg:
    type: str = "cosine"
    eta_min: float = 1e-7
    warmup_epochs: int = 0     # optionnel (0 = désactivé, comportement notebook)


@dataclass
class TrainCfg:
    epochs: int = 50
    patience: int = 10
    head_only_epochs: int = 0  # optionnel (0 = désactivé) — gèle le backbone N epochs
    amp: bool = True
    seed: int = 42
    save_every: int = 10


@dataclass
class ProbeCfg:
    C_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    max_iter: int = 2000
    knn_k: list[int] = field(default_factory=lambda: [5, 10, 20])
    include_finetuned: bool = False
    # Métrique de sélection du C (sur val), alignée par défaut sur la métrique reportée.
    selection_metric: str = "f1_macro_all"   # f1_macro_all | f1_macro_present


@dataclass
class LatentCfg:
    n_pairs: int = 10000
    rankme_subsample: int = 20000
    rankme_split: str = "test"         # RankMe et anisotropie sur le MÊME split = test
    anisotropy_split: str = "test"


@dataclass
class FeaturesCfg:
    cache: bool = True
    remap_v3: bool = True      # remappe les embeddings ancien-split -> v3 par nom de fichier
    layerwise: bool = False


# Modèles frozen évalués par défaut (probe / k-NN / latent), et fine-tunés optionnels.
_DEFAULT_MODELS = ["resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
                   "dinov3_vitl16_sat", "simdinov2_vitb16", "simdinov2_vitl16"]
_DEFAULT_FINETUNED = ["resnet50_arctic", "vitb16_arctic"]


@dataclass
class Config:
    """Configuration complète d'un run (toutes étapes confondues)."""
    env: str = "local"
    regime: str = "full"       # frozen | mhsa | full (interdit sur CNN -> erreur dans models)
    models: list[str] = field(default_factory=lambda: list(_DEFAULT_MODELS))
    finetuned_models: list[str] = field(default_factory=lambda: list(_DEFAULT_FINETUNED))
    paths: Paths = field(default_factory=Paths)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    probe: ProbeCfg = field(default_factory=ProbeCfg)
    latent: LatentCfg = field(default_factory=LatentCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    raw: dict[str, Any] = field(default_factory=dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusion récursive : les dicts sont fusionnés, le reste écrasé par ``override``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _only(cls, d: dict) -> dict:
    """Filtre/valide ``d`` contre les champs de la dataclass ``cls`` (typos -> erreur)."""
    names = {f.name for f in dc_fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: clés de config inconnues {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in names}


def load_config(path: str, base_path: str = BASE_CONFIG) -> Config:
    """Charge ``path`` fusionné par-dessus ``base.yaml`` et renvoie un :class:`Config`."""
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}
    if os.path.abspath(path) == os.path.abspath(base_path):
        merged = base
    else:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        merged = _deep_merge(base, override)
    return build_config(merged)


def build_config(d: dict) -> Config:
    """Construit un :class:`Config` à partir d'un dict fusionné, chemins résolus par ``env``."""
    env = d.get("env", "local")
    paths_block = d.get(f"paths_{env}")
    if paths_block is None:
        paths_block = d.get("paths", {})
    return Config(
        env=env,
        regime=d.get("regime", "full"),
        paths=Paths(**_only(Paths, paths_block)),
        data=DataCfg(**_only(DataCfg, d.get("data", {}))),
        model=ModelCfg(**_only(ModelCfg, d.get("model", {}))),
        optim=OptimCfg(**_only(OptimCfg, d.get("optim", {}))),
        schedule=ScheduleCfg(**_only(ScheduleCfg, d.get("schedule", {}))),
        train=TrainCfg(**_only(TrainCfg, d.get("train", {}))),
        probe=ProbeCfg(**_only(ProbeCfg, d.get("probe", {}))),
        latent=LatentCfg(**_only(LatentCfg, d.get("latent", {}))),
        features=FeaturesCfg(**_only(FeaturesCfg, d.get("features", {}))),
        raw=d,
    )
