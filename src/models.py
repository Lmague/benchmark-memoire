"""Construction des modèles : fine-tuning (régimes) + extracteurs frozen + couche-par-couche.

Régime = champ de config, jamais un fichier. Le gel/dégel et les groupes de params
(``backbone``/``attn``/``head``) sont produits ici ; les LR viennent de la config (engine).
MHSA-only n'existe que pour les ViT → ``ValueError`` claire sur un CNN.

``timm`` / ``torchvision`` / ``torch.hub`` sont importés paresseusement (jamais à
``import src``). La validation de régime est faite AVANT tout import lourd.
"""
from __future__ import annotations

CNN_NAMES = {"resnet50"}
VIT_NAMES = {"vitb16"}
# Familles non encore implémentées (points d'extension propres) :
EXTENSION_NAMES = {"dinov3_ft", "simdino_ft", "satmae"}

_TIMM_ID = {"resnet50": "resnet50", "vitb16": "vit_base_patch16_224"}


# --------------------------------------------------------------------------- régimes
def _validate_regime(name: str, regime: str) -> None:
    """Valide la compatibilité (modèle, régime) sans aucun import lourd."""
    if name in EXTENSION_NAMES:
        raise NotImplementedError(
            f"'{name}' est un point d'extension non implémenté (voir README §Extensions). "
            "DINOv3/SimDINO fine-tuning réutilisent la logique de régime ViT ; SatMAE = "
            "encodeur MAE satellite (poids + normalisation à sourcer).")
    if name in CNN_NAMES:
        if regime == "mhsa":
            raise ValueError(
                "MHSA-only n'existe que pour les ViT — il n'y a pas d'attention multi-têtes "
                f"dans un CNN. regime='mhsa' est invalide pour le modèle '{name}'.")
        if regime not in ("frozen", "full"):
            raise ValueError(f"régime '{regime}' inconnu pour '{name}' (attendu: frozen|full).")
    elif name in VIT_NAMES:
        if regime not in ("frozen", "mhsa", "full"):
            raise ValueError(f"régime '{regime}' inconnu pour '{name}' (attendu: frozen|mhsa|full).")
    else:
        raise ValueError(f"modèle de fine-tuning inconnu : '{name}'.")


def _set_requires_grad(model, predicate) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = bool(predicate(n))


def _vit_groups(model, regime: str) -> dict:
    """Applique le régime sur un ViT timm et renvoie les groupes de params entraînables."""
    named = list(model.named_parameters())
    if regime == "frozen":
        _set_requires_grad(model, lambda n: "head" in n)
        return {"head": [p for n, p in named if "head" in n]}
    if regime == "mhsa":
        _set_requires_grad(model, lambda n: ("attn" in n) or ("head" in n))
        return {
            "attn": [p for n, p in named if "attn" in n and p.requires_grad],
            "head": [p for n, p in named if "head" in n and p.requires_grad],
        }
    # full
    _set_requires_grad(model, lambda n: True)
    return {
        "backbone": [p for n, p in named if "head" not in n],
        "head": [p for n, p in named if "head" in n],
    }


def _resnet_groups(model, regime: str) -> dict:
    """Applique le régime sur un ResNet timm (head = ``fc``)."""
    named = list(model.named_parameters())
    is_head = lambda n: n.startswith("fc")  # noqa: E731
    if regime == "frozen":
        _set_requires_grad(model, is_head)
        return {"head": [p for n, p in named if is_head(n)]}
    # full
    _set_requires_grad(model, lambda n: True)
    return {
        "backbone": [p for n, p in named if not is_head(n)],
        "head": [p for n, p in named if is_head(n)],
    }


def build_model(name: str, regime: str, num_classes: int, drop_path_rate: float = 0.1):
    """Construit (model, param_groups) pour le fine-tuning.

    ``param_groups`` : dict ``{nom_groupe: [Parameters]}`` ; les LR sont appliqués par
    :func:`engine.build_optimizer` à partir de ``config.optim.lr`` (mêmes noms de groupes).
    """
    name = name.lower()
    _validate_regime(name, regime)  # peut lever AVANT tout import lourd
    import timm

    if name in CNN_NAMES:
        model = timm.create_model(_TIMM_ID[name], pretrained=True, num_classes=num_classes)
        return model, _resnet_groups(model, regime)

    # ViT (timm)
    model = timm.create_model(_TIMM_ID[name], pretrained=True, num_classes=num_classes,
                              drop_path_rate=drop_path_rate)
    return model, _vit_groups(model, regime)


# --------------------------------------------------------------- extracteurs frozen
def _forward_direct(model, x):
    """Forward standard : la sortie est déjà la représentation (CLS / pooled)."""
    return model(x)


def _dinov3_hf_forward(model, x):
    """Extrait le pooler_output d'un DINOv3 HuggingFace (AutoModel)."""
    return model(pixel_values=x).pooler_output


def _simdino_forward(model, x):
    """Extrait le token CLS d'un ViT DINOv2-with-registers (SimDINOv2).

    Référence : sslplant/simdinov2/eval/eval.py utilise ``feats = encoder(imgs)['x_norm_clstoken']``.
    On privilégie ``forward_features(x)['x_norm_clstoken']`` (API DINOv2 canonique), avec repli
    sur la logique ``cls_forward`` du notebook (dict / tuple / (B,tokens,dim)->[:,0] / (B,dim)).
    """
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict) and "x_norm_clstoken" in out:
            return out["x_norm_clstoken"]
    out = model(x)
    if isinstance(out, dict):
        return out.get("x_norm_clstoken", out.get("x_norm_cls_token"))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        out = out[:, 0]
    return out


def build_frozen_extractor(name: str, checkpoint: str | None = None):
    """Charge un backbone frozen pour extraction de features.

    Retourne ``(model, forward_fn, embed_dim, norm_key)``. Les loaders/poids reproduisent
    les notebooks d'extraction. ``checkpoint`` est requis pour SimDINOv2 (chemin du .pth teacher).
    """
    name = name.lower()
    if name == "resnet50_imagenet":
        import torch.nn as nn
        import torchvision.models as tvm
        m = tvm.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Identity()
        return m, _forward_direct, 2048, "imagenet"
    if name == "vitb16_imagenet":
        import torch.nn as nn
        import torchvision.models as tvm
        m = tvm.vit_b_16(weights="IMAGENET1K_V1")
        m.heads = nn.Identity()
        return m, _forward_direct, 768, "imagenet"
    if name == "dinov3_vitb16_lvd":
        # HuggingFace (reproductibilité notebook legacy) : transformers.AutoModel + pooler_output
        # Réf. : facebook/dinov3-vitb16-pretrain-lvd1689m, dim=768, norm ImageNet
        # (le CDN torch.hub facebookresearch/dinov3 renvoie 403 Forbidden sur Colab)
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("dinov3_vitb16_lvd nécessite transformers>=4.56.0")
        m = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
        return m, _dinov3_hf_forward, 768, "imagenet"
    if name == "dinov3_vitl16_lvd":
        # HuggingFace (reproductibilité notebook legacy) : transformers.AutoModel + pooler_output
        # Réf. : facebook/dinov3-vitl16-pretrain-lvd1689m, dim=1024, norm ImageNet
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "dinov3_vitl16_lvd nécessite transformers>=4.56.0 : pip install transformers")
        m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        return m, _dinov3_hf_forward, 1024, "imagenet"
    if name == "dinov3_vitl16_sat":
        # HuggingFace (reproductibilité notebook legacy) : transformers.AutoModel + pooler_output
        # Réf. : facebook/dinov3-vitl16-pretrain-sat493m, dim=1024, norm satellite dédiée
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "dinov3_vitl16_sat nécessite transformers>=4.56.0 : pip install transformers")
        m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
        return m, _dinov3_hf_forward, 1024, "dinov3_sat"
    if name in ("simdinov2_vitb16", "simdinov2_vitl16"):
        arch = "vitb" if name.endswith("b16") else "vitl"
        dim = 768 if arch == "vitb" else 1024
        if not checkpoint:
            raise ValueError(
                f"{name} : aucun checkpoint fourni. Renseigne le champ `checkpoint:` "
                f"(chemin du .pth teacher SimDINOv2) dans configs/frozen_{name}.yaml.")
        model = _load_simdinov2(arch, checkpoint)
        return model, _simdino_forward, dim, "simdino_inat"
    if name == "satmae":
        raise NotImplementedError(
            "SatMAE est un point d'extension (voir README §Extensions) : encodeur MAE "
            "satellite, poids + normalisation dédiée à sourcer puis ajouter ici.")
    raise ValueError(f"extracteur frozen inconnu : '{name}'.")


_SSLPLANT_REPO = "https://github.com/ilyassmoummad/sslplant.git"


def _ensure_sslplant_on_path() -> str:
    """Clone sslplant dans ``<racine>/vendors/sslplant`` si absent et l'ajoute au sys.path.

    Fait UNIQUEMENT à l'appel (jamais à l'import du module), pour préserver ``import src``.
    Ajoute ``simdinov2/`` et ``simdinov2/eval/`` au path car ``get_model.py`` (dans eval/)
    fait ``from models import vit_base`` (models = simdinov2/models, layout du repo).
    """
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # racine du package
    repo = os.path.join(root, "vendors", "sslplant")
    if not os.path.isdir(repo):
        os.makedirs(os.path.dirname(repo), exist_ok=True)
        print(f"[simdinov2] clone {_SSLPLANT_REPO} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", _SSLPLANT_REPO, repo], check=True)
    sim = os.path.join(repo, "simdinov2")
    for p in (sim, os.path.join(sim, "eval")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return repo


def _load_simdinov2(arch: str, checkpoint: str):
    """Charge un encodeur SimDINOv2 (``arch`` ∈ {vitb, vitl}) depuis un .pth teacher.

    Réutilise le loader officiel du repo (``simdinov2/eval/get_model.py``) :
    ``initialize_encoder`` (vit_base/vit_large DINOv2-with-registers) + ``load_simdino_state_dict``
    (clé ``teacher`` → suppression du préfixe ``backbone.`` → ``remap_key``). On charge en
    ``strict=False`` et on lève si >10% des clés du modèle manquent (sécurité ; le repo, lui,
    utilise strict=True sur un dict déjà nettoyé).
    """
    import os
    import torch
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint SimDINOv2 introuvable : {checkpoint}")
    _ensure_sslplant_on_path()
    from get_model import initialize_encoder, load_simdino_state_dict  # simdinov2/eval/get_model.py

    encoder = initialize_encoder(arch=arch)
    state_dict = load_simdino_state_dict(checkpoint)
    incompatible = encoder.load_state_dict(state_dict, strict=False)
    n_model = len(encoder.state_dict())
    n_missing = len(incompatible.missing_keys)
    frac = n_missing / max(1, n_model)
    print(f"[simdinov2] {arch}: chargé {n_model - n_missing}/{n_model} clés "
          f"(manquantes={n_missing}, inattendues={len(incompatible.unexpected_keys)})")
    if frac > 0.10:
        raise RuntimeError(
            f"SimDINOv2 {arch} : {n_missing}/{n_model} clés du modèle manquantes "
            f"({frac:.0%} > 10%) — checkpoint/architecture incompatibles "
            f"(clés inattendues={len(incompatible.unexpected_keys)}).")
    return encoder.eval()


# --------------------------------------------------------- accès couche-par-couche
def _is_module_list(obj) -> bool:
    """Vrai si ``obj`` est une séquence de sous-modules (ModuleList/Sequential, len>=2)."""
    try:
        return len(obj) >= 2
    except TypeError:
        return False


def get_transformer_blocks(model):
    """Retourne la ModuleList des blocs transformer pour l'extraction couche-par-couche.

    Couvre les conventions d'attributs connues :
      - timm ViT / DINOv3 du hub : ``model.blocks`` ;
      - torchvision ViT          : ``model.encoder.layers`` ;
      - HuggingFace DINOv2       : ``model.encoder.layer`` (singulier) ;
      - HuggingFace DINOv3 (``Dinov3ViTModel``, transformers>=4.56) : ``model.layer``.

    L'attribut exact de DINOv3-HF n'a pas pu être vérifié hors-ligne (transformers non
    installé sur la machine d'analyse) : on tente donc plusieurs chemins, puis, en dernier
    recours, on prend la plus longue ``nn.ModuleList`` du modèle. Le garde-fou de
    :func:`extract_layerwise` valide ensuite que ``len(blocks) == nb attendu`` (24 pour ViT-L).
    Lève ``ValueError`` pour un modèle sans blocs (ex. CNN).
    """
    for attr_path in (("blocks",), ("layer",), ("layers",),
                      ("encoder", "layers"), ("encoder", "layer")):
        obj = model
        ok = True
        for a in attr_path:
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok and _is_module_list(obj):
            return obj
    # Repli robuste (structure HF inattendue) : la plus longue ModuleList du modèle.
    import torch.nn as nn
    longest = None
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) >= 2:
            if longest is None or len(m) > len(longest):
                longest = m
    if longest is not None:
        return longest
    raise ValueError(
        "Blocs transformer introuvables : l'extraction couche-par-couche n'est supportée "
        "que pour les ViT (timm/torchvision/DINOv3/HF), pas pour les CNN.")
