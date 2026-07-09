"""Extraction et cache des représentations frozen (CLS/pooled) + remap split + couche-par-couche.

- Cache fp16 ``.npy`` par split, avec cache-hit (réutilisé par probe, k-NN et analyse latente).
- Remap des embeddings (ordre ancien split) vers le split v3 PAR NOM DE FICHIER.
- Extraction couche-par-couche : représentation CLS à la sortie de chaque bloc transformer.

Volontairement importable sans torch/torchvision : seuls numpy + stdlib au niveau module ;
les dépendances lourdes sont importées dans les fonctions d'extraction (GPU/Colab).
``remap_to_v3`` / ``load_features`` n'ont besoin que de numpy → utilisables en local.
"""
from __future__ import annotations

import os

import numpy as np

from .utils import ensure_dir, is_sota_key, read_split_csv, sota_regime

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------- chemins de cache
def _emb_path(emb_dir: str, key: str, split: str) -> str:
    return os.path.join(emb_dir, f"{key}_{split}.npy")


def _lbl_path(emb_dir: str, key: str, split: str) -> str:
    return os.path.join(emb_dir, f"{key}_{split}_labels.npy")


def _layer_emb_path(emb_dir: str, key: str, split: str, layer: int) -> str:
    return os.path.join(emb_dir, f"{key}_{split}_layer{layer:02d}.npy")


def _infer_expected_blocks(model_key: str):
    k = model_key.lower()
    if "vitl" in k:
        return 24
    if "vitb" in k:
        return 12
    return None


def count_layerwise_layers(cfg, model_key: str, split: str) -> int:
    import glob
    pattern = os.path.join(cfg.paths.emb_dir, f"{model_key}_{split}_layer*.npy")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier layerwise pour {model_key}/{split} dans {cfg.paths.emb_dir}. "
            f"Lancer extract.py --layerwise d'abord.")
    indices = []
    for f in files:
        base = os.path.basename(f)
        part = base.replace(f"{model_key}_{split}_layer", "").replace(".npy", "")
        try:
            indices.append(int(part))
        except ValueError:
            continue
    indices = sorted(indices)
    if indices != list(range(len(indices))):
        raise RuntimeError(
            f"Trous dans les couches layerwise {model_key}/{split} : {indices}")
    return len(indices)


# --------------------------------------------------------------- loaders d'extraction
def _eval_loader(cfg, split: str, mean, std):
    """DataLoader déterministe (transforms d'éval, normalisation du modèle)."""
    from torch.utils.data import DataLoader

    from .data import ArcticTVCDataset, build_transforms
    tf = build_transforms("eval", mean, std, cfg.data.image_size)
    ds = ArcticTVCDataset(os.path.join(cfg.paths.csv_dir, f"{split}.csv"),
                          cfg.paths.tiles_dir, tf)
    return DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers, pin_memory=True)


def sanity_check(model, forward_fn, cfg, norm_key: str, n: int = 8, split: str = "val") -> tuple:
    """Diagnostic sur ``n`` images AVANT l'extraction des 80k : shape (B,dim), std/dim, ||v||.

    Lève ``RuntimeError`` si la sortie n'est pas ``(n, dim)``, si ``std/dim <= 0.01`` (collapse /
    CLS mal extrait) ou si ``||v|| <= 1.0``. Repris du ``sanity_check`` de
    ``extract_embeddings.ipynb`` (source de vérité). Nécessite torch + tuiles.
    """
    import torch

    from .data import ArcticTVCDataset, build_transforms
    from .utils import get_device, get_normalization
    mean, std = get_normalization(norm_key)
    ds = ArcticTVCDataset(os.path.join(cfg.paths.csv_dir, f"{split}.csv"), cfg.paths.tiles_dir,
                          build_transforms("eval", mean, std, cfg.data.image_size))
    device = get_device()
    xb = torch.stack([ds[i][0] for i in range(n)]).to(device)
    model.eval()
    with torch.no_grad():
        out = forward_fn(model, xb).float().cpu()
    std_dim = float(out.std(0).mean()) if out.ndim == 2 else -1.0
    nrm = float(out.norm(dim=1).mean()) if out.ndim == 2 else -1.0
    print(f"[sanity] shape={tuple(out.shape)} std/dim={std_dim:.4f} ||v||={nrm:.2f}")
    problems = []
    if out.ndim != 2 or out.shape[0] != n:
        problems.append(f"shape {tuple(out.shape)} (attendu ({n}, dim))")
    if std_dim <= 0.01:
        problems.append(f"std/dim={std_dim:.4f} <= 0.01 (collapse / CLS mal extrait)")
    if nrm <= 1.0:
        problems.append(f"||v||={nrm:.2f} <= 1.0")
    if problems:
        raise RuntimeError("[sanity] échec: " + " ; ".join(problems))
    return tuple(out.shape)


# -------------------------------------------------------------- extraction frozen
def extract_features(cfg, model_key: str, splits=SPLITS) -> dict:
    """Extrait + cache (fp16) les features frozen de ``model_key``. Cache-hit par split.

    Retourne ``{split: (E float16, L int64)}``. Nécessite torch/torchvision (Colab/GPU).
    """
    import torch

    from .models import build_frozen_extractor
    from .utils import get_device, get_normalization
    emb_dir = cfg.paths.emb_dir
    ensure_dir(emb_dir)

    out: dict = {}
    todo: list[str] = []
    for s in splits:
        ep, lp = _emb_path(emb_dir, model_key, s), _lbl_path(emb_dir, model_key, s)
        if cfg.features.cache and os.path.exists(ep) and os.path.exists(lp):
            out[s] = (np.load(ep), np.load(lp))
            print(f"[extract] cache-hit {model_key}/{s}")
        else:
            todo.append(s)
    if not todo:
        return out

    model, forward_fn, dim, norm_key = build_frozen_extractor(model_key, cfg.raw.get("checkpoint"))
    device = get_device()
    model = model.to(device).eval()
    sanity_check(model, forward_fn, cfg, norm_key)   # diagnostic 8 images avant les 80k
    mean, std = get_normalization(norm_key)
    for s in todo:
        loader = _eval_loader(cfg, s, mean, std)
        embs, lbls = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                f = forward_fn(model, x)
                embs.append(f.detach().cpu().to(torch.float16).numpy())
                lbls.append(y.numpy())
        E = np.concatenate(embs, axis=0)
        L = np.concatenate(lbls, axis=0)
        np.save(_emb_path(emb_dir, model_key, s), E)
        np.save(_lbl_path(emb_dir, model_key, s), L)
        out[s] = (E, L)
        print(f"[extract] {model_key}/{s}: {E.shape} (dim={dim}) -> cache")
    return out


# ----------------------------------------------------- extraction couche-par-couche
def _layerwise_sanity(model, forward_fn, blocks, sel, cfg, norm_key, dim, model_key,
                      split: str, n: int = 8) -> None:
    """Garde-fou AVANT l'extraction des 80k : valide la structure layerwise sur ``n`` tuiles.

    Lève ``RuntimeError`` immédiatement si :
      - ``len(blocks)`` != nb de blocs attendu pour l'archi (ViT-L/16=24, ViT-B/16=12) ;
      - la DERNIÈRE couche n'émet pas un tenseur ``(B, tokens, dim)`` avec la bonne ``dim`` ;
      - ``std/dim <= 0.01`` sur le CLS (``o[:, 0, :]``) de la DERNIÈRE couche : collapse / hook
        mal placé. Le seuil n'est PAS appliqué aux couches basses (CLS peu discriminant en bloc
        0/1, std/dim≈0.006 : comportement normal, pas un collapse).
    """
    import torch

    from .data import ArcticTVCDataset, build_transforms
    from .utils import get_device, get_normalization

    expected = _infer_expected_blocks(model_key)
    if expected is not None and len(blocks) != expected:
        raise RuntimeError(
            f"[layerwise:sanity] {model_key}: {len(blocks)} blocs captés via "
            f"get_transformer_blocks, {expected} attendus pour cette architecture — "
            "mauvaise ModuleList (structure HF inattendue ?). Extraction NON lancée.")

    mean, std = get_normalization(norm_key)
    ds = ArcticTVCDataset(os.path.join(cfg.paths.csv_dir, f"{split}.csv"), cfg.paths.tiles_dir,
                          build_transforms("eval", mean, std, cfg.data.image_size))
    device = get_device()
    xb = torch.stack([ds[i][0] for i in range(n)]).to(device)

    raw: dict[int, "torch.Tensor"] = {}
    handles = []
    for li in sel:
        def _h(_m, _inp, out, li=li):
            raw[li] = (out[0] if isinstance(out, tuple) else out).detach()
        handles.append(blocks[li].register_forward_hook(_h))
    try:
        model.eval()
        with torch.no_grad():
            forward_fn(model, xb)
    finally:
        for h in handles:
            h.remove()

    # Seuil std/dim sur la DERNIÈRE couche seulement (les couches basses ont un CLS
    # naturellement peu discriminant : ce n'est pas un collapse). Structure validée sur ce bloc.
    last = sel[-1]
    o = raw.get(last)
    if o is None:
        raise RuntimeError(f"[layerwise:sanity] bloc {last}: aucun tenseur capturé (hook inactif).")
    if o.ndim != 3 or o.shape[0] != n:
        raise RuntimeError(
            f"[layerwise:sanity] bloc {last}: forme {tuple(o.shape)} (attendu ({n}, tokens, dim)).")
    if o.shape[-1] != dim:
        raise RuntimeError(
            f"[layerwise:sanity] bloc {last}: dim={o.shape[-1]} ≠ {dim} attendu.")
    sd = float(o[:, 0, :].float().cpu().std(0).mean())
    if sd <= 0.01:
        raise RuntimeError(
            f"[layerwise:sanity] bloc {last}: std/dim={sd:.4f} <= 0.01 sur le CLS "
            "(collapse / hook mal placé).")
    print(f"[layerwise:sanity] OK {model_key}: {len(blocks)} blocs, dim={dim}, "
          f"CLS dernière couche (bloc {last}) validé sur {n} tuiles (split={split}).")


def extract_layerwise(cfg, model_key: str, splits=SPLITS, layers=None) -> int:
    """Extrait + cache la représentation CLS à la sortie de chaque bloc transformer.

    Cache ``{key}_{split}_layer{idx}.npy`` (fp16) + labels partagés. Retourne le nombre
    de couches. Nécessite torch (ViT uniquement ; CNN -> ValueError via get_transformer_blocks).
    """
    import torch

    from .models import build_frozen_extractor, get_transformer_blocks
    from .utils import get_device, get_normalization
    emb_dir = cfg.paths.emb_dir
    ensure_dir(emb_dir)

    model, forward_fn, dim, norm_key = build_frozen_extractor(model_key, cfg.raw.get("checkpoint"))
    device = get_device()
    model = model.to(device).eval()
    blocks = get_transformer_blocks(model)
    sel = list(range(len(blocks))) if layers is None else list(layers)

    # Garde-fou : valide structure/dim/CLS sur 8 tuiles AVANT la boucle sur les 80k.
    _layerwise_sanity(model, forward_fn, blocks, sel, cfg, norm_key, dim, model_key,
                      split=splits[0])

    captured: dict[int, "torch.Tensor"] = {}
    handles = []
    for li in sel:
        def _hook(_m, _inp, out, li=li):
            o = out[0] if isinstance(out, tuple) else out
            captured[li] = o[:, 0, :].detach().cpu().to(torch.float16)  # token CLS

        handles.append(blocks[li].register_forward_hook(_hook))

    mean, std = get_normalization(norm_key)
    try:
        for s in splits:
            loader = _eval_loader(cfg, s, mean, std)
            per_layer = {li: [] for li in sel}
            lbls = []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    captured.clear()
                    forward_fn(model, x)
                    for li in sel:
                        per_layer[li].append(captured[li].numpy())
                    lbls.append(y.numpy())
            L = np.concatenate(lbls, axis=0)
            np.save(_lbl_path(emb_dir, f"{model_key}_layerwise", s), L)
            for li in sel:
                E = np.concatenate(per_layer[li], axis=0)
                np.save(_layer_emb_path(emb_dir, model_key, s, li), E)
            print(f"[layerwise] {model_key}/{s}: {len(sel)} couches × {L.shape[0]} ex")
    finally:
        for h in handles:
            h.remove()
    return len(sel)


def load_layerwise(cfg, model_key: str, split: str, n_layers: int) -> dict:
    """Charge les features couche-par-couche cachées : ``{layer_idx: (E float32, L int64)}``."""
    emb_dir = cfg.paths.emb_dir
    L = np.load(_lbl_path(emb_dir, f"{model_key}_layerwise", split)).astype(np.int64)
    out = {}
    for li in range(n_layers):
        E = np.load(_layer_emb_path(emb_dir, model_key, split, li)).astype(np.float32)
        out[li] = (E, L)
    return out


# ------------------------------------------------------------------ remap / chargement
def remap_to_v3(cfg, model_key: str, splits=SPLITS) -> dict:
    """Remappe les embeddings cachés (ordre ancien split) vers v3 par nom de fichier.

    Reproduit ``latent_v3_from_embeddings.load_model_v3`` : indexe filepath -> (emb, label)
    sur l'ancien split, puis ré-assemble selon les CSV v3. Assertion labels v3 == labels stockés.
    """
    emb_dir, old_dir, v3_dir = cfg.paths.emb_dir, cfg.paths.old_csv_dir, cfg.paths.csv_dir
    fp2emb: dict[str, np.ndarray] = {}
    fp2lab: dict[str, int] = {}
    for s in splits:
        fps, _ = read_split_csv(os.path.join(old_dir, f"{s}.csv"))
        E = np.load(_emb_path(emb_dir, model_key, s))
        L = np.load(_lbl_path(emb_dir, model_key, s))
        assert len(fps) == E.shape[0] == L.shape[0], f"désalignement {model_key}/{s}"
        for i, fp in enumerate(fps):
            fp2emb[fp] = E[i]
            fp2lab[fp] = int(L[i])
    out = {}
    for s in splits:
        fps, lbs = read_split_csv(os.path.join(v3_dir, f"{s}.csv"))
        E = np.stack([fp2emb[fp] for fp in fps]).astype(np.float32)
        L = np.asarray([fp2lab[fp] for fp in fps], dtype=np.int64)
        assert (L == lbs).all(), f"label mismatch {model_key}/{s}"
        out[s] = (E, L)
    return out


def _sota_run_dir(cfg, model_key: str) -> str:
    """Dossier d'embeddings d'un run SOTA : ``{sota_dir}/{regime}/embeddings/{key}``."""
    return os.path.join(cfg.paths.sota_dir, sota_regime(model_key), "embeddings", model_key)


def load_sota_features(cfg, model_key: str, splits=SPLITS) -> dict:
    """Charge les features d'un run SOTA (fichiers NUS ``{split}.npy`` / ``{split}_labels.npy``).

    Convention de nommage DIFFÉRENTE des canoniques : un dossier par run
    (``{sota_dir}/{regime}/embeddings/vitb16_{regime}_frac{XXX}_seed{N}/``) contenant
    ``train.npy``/``val.npy``/``test.npy`` + labels, SANS préfixe modèle. Labels déjà en
    schéma 11cls_no_rhol (0–10). Aucun remap v3 (extraits directement en ordre v3).
    """
    run_dir = _sota_run_dir(cfg, model_key)
    out = {}
    for s in splits:
        ep = os.path.join(run_dir, f"{s}.npy")
        lp = os.path.join(run_dir, f"{s}_labels.npy")
        E = np.load(ep).astype(np.float32)
        L = np.load(lp).astype(np.int64)
        out[s] = (E, L)
    return out


def load_features(cfg, model_key: str, splits=SPLITS) -> dict:
    """Charge les features d'un modèle.

    - Runs SOTA (``is_sota_key``) : dossier-par-run, fichiers nus (schéma 11cls).
    - Canoniques : remappés v3 si ``cfg.features.remap_v3`` sinon directs (schéma 12cls).
    """
    if is_sota_key(model_key):
        return load_sota_features(cfg, model_key, splits)
    if cfg.features.remap_v3:
        return remap_to_v3(cfg, model_key, splits)
    out = {}
    for s in splits:
        E = np.load(_emb_path(cfg.paths.emb_dir, model_key, s)).astype(np.float32)
        L = np.load(_lbl_path(cfg.paths.emb_dir, model_key, s)).astype(np.int64)
        out[s] = (E, L)
    return out
