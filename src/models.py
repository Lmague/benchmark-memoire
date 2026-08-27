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
# Backbones SSL frozen (build_frozen_extractor) rendus fine-tunables via un wrapper
# nn.Module {backbone, head} — régimes frozen|mhsa|full|explora_like (pas de 'scratch' :
# init aléatoire n'a pas de sens pour un backbone pré-entraîné).
SSL_FT_NAMES = {"dinov3_vits16_lvd", "dinov3_vitb16_lvd", "dinov3_vitl16_lvd", "dinov3_vith16plus_lvd", "simdinov2_vitb16", "simdinov2_vitl16"}

_TIMM_ID = {"resnet50": "resnet50", "vitb16": "vit_base_patch16_224"}


# --------------------------------------------------------------------------- régimes
def _validate_regime(name: str, regime: str) -> None:
    """Valide la compatibilité (modèle, régime) sans aucun import lourd."""
    if name in CNN_NAMES:
        if regime == "mhsa":
            raise ValueError(
                "MHSA-only n'existe que pour les ViT — il n'y a pas d'attention multi-têtes "
                f"dans un CNN. regime='mhsa' est invalide pour le modèle '{name}'.")
        if regime not in ("frozen", "full"):
            raise ValueError(f"régime '{regime}' inconnu pour '{name}' (attendu: frozen|full).")
    elif name in VIT_NAMES:
        if regime not in ("frozen", "mhsa", "full", "explora_like", "scratch", "lora"):
            raise ValueError(
                f"régime '{regime}' inconnu pour '{name}' "
                f"(attendu: frozen|mhsa|full|explora_like|scratch).")
    elif name in SSL_FT_NAMES:
        if regime not in ("frozen", "mhsa", "full", "explora_like", "lora"):
            raise ValueError(
                f"régime '{regime}' inconnu pour '{name}' (attendu: frozen|mhsa|full|explora_like — "
                "scratch non supporté pour les backbones SSL).")
    else:
        raise ValueError(f"modèle de fine-tuning inconnu : '{name}'.")


def _set_requires_grad(model, predicate) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = bool(predicate(n))


def _is_attn_param(n: str) -> bool:
    """Nom de paramètre d'attention multi-têtes, toutes conventions supportées :

    timm/DINOv2 (``blocks.N.attn.*``) et HuggingFace DINOv3 (``model.layer.N.attention.*``).
    Superset sûr : aucun paramètre timm existant ne contient la sous-chaîne "attention".
    """
    return ("attn" in n) or ("attention" in n)


def _vit_groups(model, regime: str) -> dict:
    """Applique le régime sur un ViT (timm, ou wrapper SSL {backbone,head}) et renvoie
    les groupes de params entraînables. Le prédicat d'attention couvre timm/DINOv2
    ("attn") et HuggingFace DINOv3 ("attention") — cf. :func:`_is_attn_param`.
    """
    named = list(model.named_parameters())
    if regime == "frozen":
        _set_requires_grad(model, lambda n: "head" in n)
        return {"head": [p for n, p in named if "head" in n]}
    if regime == "mhsa":
        _set_requires_grad(model, lambda n: _is_attn_param(n) or ("head" in n))
        return {
            "attn": [p for n, p in named if _is_attn_param(n) and p.requires_grad],
            "head": [p for n, p in named if "head" in n and p.requires_grad],
        }
    # full
    _set_requires_grad(model, lambda n: True)
    return {
        "backbone": [p for n, p in named if "head" not in n],
        "head": [p for n, p in named if "head" in n],
    }


# ----------------------------------------------------- LoRA (régime explora_like)
# LoRA minimal maison (peft non requis) : low-rank A·B sur les projections Q/V de
# l'attention. Le papier ExPLoRA (arXiv:2406.10973) applique LoRA sur Q,V uniquement.
# Deux architectures d'attention coexistent dans ce projet, donc DEUX classes LoRA :
#   - timm ViT (qkv FUSIONNÉ, une seule Linear dim -> 3·dim)      -> LoRAFusedQKV
#     (slice de sortie ciblée, K laissé intact si non ciblé) ;
#   - HuggingFace DINOv3 (q_proj/k_proj/v_proj/o_proj SÉPARÉS,    -> LoRALinear
#     vérifié en inspectant transformers.models.dinov3_vit : pas de qkv fusionné)
#     chaque projection ciblée reçoit son propre A/B, pas de slice à gérer.
# Le delta est FUSIONNÉ dans le(s) poids de base à l'extraction (merge_lora_state_dict) :
# le checkpoint redevient un modèle standard (timm ou HF), agnostique au régime.
_LORA_CLASS = None
_QKV_SLICE = {"q": 0, "k": 1, "v": 2}  # ordre de fusion timm : [Q | K | V]
_PROJ_ATTR = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}  # HF DINOv3 : projections séparées


def _get_lora_class():
    """Définit/retourne la classe ``LoRAFusedQKV`` (lazy : préserve ``import src.models`` sans torch)."""
    global _LORA_CLASS
    if _LORA_CLASS is not None:
        return _LORA_CLASS
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRAFusedQKV(nn.Module):
        """Wrappe un ``nn.Linear`` qkv fusionné (timm ViT) + adaptateurs LoRA par slice.

        Base (``weight``/``bias``) GELÉE ; seuls ``lora_A[t]``/``lora_B[t]`` (t ∈ targets)
        s'entraînent. ``scaling = alpha/r`` stocké en buffer pour la fusion hors-ligne.
        Init standard LoRA : A ~ kaiming_uniform, B = 0 (delta nul au départ → init ImageNet
        préservée). Forward identique en shape à la Linear d'origine → drop-in pour timm.
        """

        def __init__(self, base, r, alpha, targets=("q", "v"), dropout=0.0):
            super().__init__()
            self.in_features = base.in_features
            self.out_features = base.out_features        # 3·dim
            self.dim = self.out_features // 3
            self.r = int(r)
            self.scaling = float(alpha) / float(r)
            self.targets = tuple(t.lower() for t in targets)
            self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
            if base.bias is not None:
                self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            else:
                self.register_parameter("bias", None)
            self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for t in self.targets:
                if t not in _QKV_SLICE:
                    raise ValueError(f"target LoRA inconnue '{t}' (attendu: q|k|v).")
                A = nn.Parameter(torch.empty(self.r, self.in_features))
                B = nn.Parameter(torch.zeros(self.dim, self.r))
                nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                self.lora_A[t] = A
                self.lora_B[t] = B
            self.register_buffer("scaling_buf", torch.tensor(self.scaling, dtype=torch.float32))

        def forward(self, x):
            out = F.linear(x, self.weight, self.bias)    # [..., 3·dim]
            xd = self.drop(x)
            for t in self.targets:
                s = _QKV_SLICE[t]
                lo, hi = s * self.dim, (s + 1) * self.dim
                d = self.scaling * F.linear(F.linear(xd, self.lora_A[t]), self.lora_B[t])
                out = out + F.pad(d, (lo, self.out_features - hi))   # fonctionnel (pas d'in-place)
            return out

    _LORA_CLASS = LoRAFusedQKV
    return _LORA_CLASS


_LORA_LINEAR_CLASS = None
_LORA_LINEAR_TARGET = "d"  # clé unique (delta plein, pas de slice) — distingue de _QKV_SLICE


def _get_lora_linear_class():
    """Définit/retourne ``LoRALinear`` (lazy, cf. :func:`_get_lora_class`).

    Contrepartie de ``LoRAFusedQKV`` pour les architectures où Q/K/V sont des ``nn.Linear``
    SÉPARÉES (HuggingFace DINOv3 : ``attention.q_proj``/``k_proj``/``v_proj``/``o_proj``,
    aucun ``qkv`` fusionné — vérifié dans ``transformers.models.dinov3_vit.modeling_dinov3_vit``).
    Un ``LoRALinear`` remplace UNE projection ciblée (ex. ``q_proj``) : pas de slice de sortie
    à gérer (contrairement à ``LoRAFusedQKV``), le delta bas-rang s'applique à toute la sortie.
    """
    global _LORA_LINEAR_CLASS
    if _LORA_LINEAR_CLASS is not None:
        return _LORA_LINEAR_CLASS
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        """Wrappe un ``nn.Linear`` standalone (base GELÉE) + un adaptateur LoRA bas-rang."""

        def __init__(self, base, r, alpha, dropout=0.0):
            super().__init__()
            self.in_features = base.in_features
            self.out_features = base.out_features
            self.r = int(r)
            self.scaling = float(alpha) / float(r)
            self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
            if base.bias is not None:
                self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            else:
                self.register_parameter("bias", None)
            self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
            A = nn.Parameter(torch.empty(self.r, self.in_features))
            B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.lora_A = nn.ParameterDict({_LORA_LINEAR_TARGET: A})
            self.lora_B = nn.ParameterDict({_LORA_LINEAR_TARGET: B})
            self.register_buffer("scaling_buf", torch.tensor(self.scaling, dtype=torch.float32))

        def forward(self, x):
            out = F.linear(x, self.weight, self.bias)
            xd = self.drop(x)
            t = _LORA_LINEAR_TARGET
            d = self.scaling * F.linear(F.linear(xd, self.lora_A[t]), self.lora_B[t])
            return out + d

    _LORA_LINEAR_CLASS = LoRALinear
    return _LORA_LINEAR_CLASS


def merge_lora_state_dict(state: dict) -> dict:
    """Fusionne tout adaptateur LoRA d'un ``state_dict`` dans le(s) poids de base correspondants.

    Pour chaque préfixe ``P`` portant des clés ``P.lora_A.<t>``/``P.lora_B.<t>`` :

    - ``qkv`` FUSIONNÉ (timm, ``t`` ∈ q/k/v) : ``W[slice_t] += scaling · (B·A)`` sur
      ``P.weight`` (slice déduite de ``<t>``, cf. :data:`_QKV_SLICE`) ;
    - projection SÉPARÉE (HF DINOv3, ``t`` == :data:`_LORA_LINEAR_TARGET`) : ``W += scaling ·
      (B·A)`` sur tout ``P.weight`` (pas de slice — une seule Linear par préfixe).

    Scaling lu dans ``P.scaling_buf``. SUPPRIME ensuite les clés LoRA et le buffer. Le
    state_dict résultant est celui d'un modèle standard (timm ou HF), sans trace de LoRA.
    No-op si aucune clé LoRA (checkpoints full/mhsa/scratch ou backbones fine-tunés legacy) →
    sûr à appeler inconditionnellement à l'extraction.
    """
    prefixes = sorted({k.split(".lora_A.")[0] for k in state if ".lora_A." in k})
    if not prefixes:
        return state
    out = dict(state)
    for p in prefixes:
        wkey = f"{p}.weight"
        if wkey not in out:
            continue
        W = out[wkey].clone()
        skey = f"{p}.scaling_buf"
        scaling = float(out[skey]) if skey in out else 1.0
        ka_full, kb_full = f"{p}.lora_A.{_LORA_LINEAR_TARGET}", f"{p}.lora_B.{_LORA_LINEAR_TARGET}"
        if ka_full in out and kb_full in out:
            A = out[ka_full].to(W.dtype)
            B = out[kb_full].to(W.dtype)
            W += scaling * (B @ A)
        else:
            dim = W.shape[0] // 3
            for t, s in _QKV_SLICE.items():
                ka, kb = f"{p}.lora_A.{t}", f"{p}.lora_B.{t}"
                if ka in out and kb in out:
                    A = out[ka].to(W.dtype)
                    B = out[kb].to(W.dtype)
                    W[s * dim:(s + 1) * dim, :] += scaling * (B @ A)
        out[wkey] = W
        for kk in list(out):
            if kk.startswith(f"{p}.lora_A.") or kk.startswith(f"{p}.lora_B.") or kk == skey:
                del out[kk]
    return out


def _find_module_name(model, target) -> str:
    """Nom pointé (``named_modules``) du sous-module ``target`` dans ``model`` (par identité)."""
    for n, m in model.named_modules():
        if m is target:
            return n
    raise ValueError("module introuvable par identité dans l'arbre (get_transformer_blocks a "
                     "renvoyé une ModuleList qui n'appartient pas à ce modèle ?).")


def _explora_groups(model, lora) -> dict:
    """Régime ``explora_like`` : LoRA Q,V sur TOUS les blocs + full-FT sur blocs U={1,L}.

    Couvre DEUX architectures d'attention (cf. :func:`_get_lora_class`/:func:`_get_lora_linear_class`) :
    ViT timm (``attn.qkv`` fusionné) et HuggingFace DINOv3 (``attention.{q,k,v,o}_proj`` séparés,
    aucun attribut ``blocks`` — les blocs vivent sous ``model.model.layer`` (potentiellement
    préfixé ``backbone.`` par :class:`SSLBackboneClassifier`), retrouvés via
    :func:`get_transformer_blocks`).

    - LoRA Q,V injecté sur TOUS les blocs (0..11).
    - full-FT sur ``full_ft_block_indices`` (défaut : {0, N-1} = U={1,L} du papier ExPLoRA).
      Les blocs full-FT ont LoRA + full-FT EN SUPERPOSITION (pas disjoints).
    - Toutes les LayerNorm (partout, + norm finale) : dégelées → groupe 'norm'.
    - Head : dégelée → groupe 'head'. pos_embed/cls_token/patch_embed : GELÉS (init pré-entraîné).
    Groupes retournés : 'lora' | 'full_late' | 'norm' | 'head' (LR fournis par cfg.optim.lr).
    """
    import torch.nn as nn
    if lora is None:
        raise ValueError("régime 'explora_like' : configuration LoRA absente (cfg.lora).")
    blocks = get_transformer_blocks(model)
    n_blocks = len(blocks)
    blocklist_name = _find_module_name(model, blocks)

    # full_ft_block_indices (optionnel) prend le dessus sur n_full_ft_blocks.
    # Défaut ExPLoRA (n_full_ft_blocks=2) → U={1,L} = premier + dernier bloc.
    explicit_idx = getattr(lora, "full_ft_block_indices", None)
    if explicit_idx is not None:
        full_ft_idx = sorted(set(int(i) for i in explicit_idx))
        if any(not 0 <= i < n_blocks for i in full_ft_idx):
            raise ValueError(f"full_ft_block_indices={full_ft_idx} invalide pour {n_blocks} blocs.")
    else:
        n_late = int(lora.n_full_ft_blocks)
        if not 0 <= n_late < n_blocks:
            raise ValueError(f"n_full_ft_blocks={n_late} invalide pour {n_blocks} blocs.")
        if n_late == 1:
            full_ft_idx = [n_blocks - 1]
        elif n_late == 2:
            full_ft_idx = [0, n_blocks - 1]                 # U={1,L}
        else:
            n_first = n_late // 2
            n_last = n_late - n_first
            full_ft_idx = list(range(0, n_first)) + list(range(n_blocks - n_last, n_blocks))

    # LoRA sur TOUS les blocs par défaut (y compris ceux en full-FT → superposition) ;
    # restreint à lora_block_indices si fourni (ablation de position : les blocs non
    # listés restent entièrement gelés).
    explicit_lora_idx = getattr(lora, "lora_block_indices", None)
    if explicit_lora_idx is not None:
        lora_idx = sorted(set(int(i) for i in explicit_lora_idx))
        if any(not 0 <= i < n_blocks for i in lora_idx):
            raise ValueError(f"lora_block_indices={lora_idx} invalide pour {n_blocks} blocs.")
    else:
        lora_idx = list(range(n_blocks))
    lora_cls = _get_lora_class()
    lora_linear_cls = _get_lora_linear_class()
    targets = tuple(t.lower() for t in lora.target_modules)

    # 1. Injection LoRA sur l'attention de TOUS les blocs (qkv fusionné OU projections séparées)
    for i in lora_idx:
        block = blocks[i]
        attn = getattr(block, "attn", None)
        if attn is None:
            attn = getattr(block, "attention", None)
        if attn is None:
            raise ValueError(
                f"bloc {i} : pas de sous-module 'attn'/'attention' (arch non supportée).")
        if hasattr(attn, "qkv"):
            attn.qkv = lora_cls(attn.qkv, r=lora.r, alpha=lora.alpha,
                                targets=targets, dropout=lora.dropout)
        else:
            for t in targets:
                proj_name = _PROJ_ATTR.get(t)
                if proj_name is None or not hasattr(attn, proj_name):
                    raise ValueError(
                        f"bloc {i} : cible LoRA '{t}' introuvable sur {type(attn).__name__} "
                        f"(ni 'qkv' fusionné, ni '{proj_name}' séparé).")
                setattr(attn, proj_name, lora_linear_cls(
                    getattr(attn, proj_name), r=lora.r, alpha=lora.alpha, dropout=lora.dropout))

    # 2. Noms exacts des paramètres de LayerNorm (robuste : par type de module)
    norm_param_names = set()
    for mname, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for pn, _ in m.named_parameters(recurse=False):
                norm_param_names.add(f"{mname}.{pn}" if mname else pn)
    late_prefixes = tuple(f"{blocklist_name}.{i}." for i in full_ft_idx)

    # 3. Attribution exclusive des groupes + requires_grad
    groups = {"lora": [], "full_late": [], "norm": [], "head": []}
    for n, p in model.named_parameters():
        if "head" in n:
            p.requires_grad_(True); groups["head"].append(p)
        elif (".lora_A." in n) or (".lora_B." in n):
            p.requires_grad_(True); groups["lora"].append(p)
        elif n in norm_param_names:
            p.requires_grad_(True); groups["norm"].append(p)
        elif n.startswith(late_prefixes):
            p.requires_grad_(True); groups["full_late"].append(p)
        else:
            p.requires_grad_(False)
    return groups


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


_SSL_CLASSIFIER_CLASS = None


def _get_ssl_classifier_class():
    """Définit/retourne ``SSLBackboneClassifier`` (lazy : préserve ``import src.models`` sans torch).

    Wrapper minimal ``{backbone, head}`` pour rendre fine-tunable un backbone SSL headless
    (DINOv3-HF, SimDINOv2) chargé par :func:`build_frozen_extractor`. ``forward_fn`` est LA
    MÊME fonction que celle utilisée pour l'extraction frozen (``_dinov3_hf_forward`` /
    ``_simdino_forward``) — garantit que la représentation fine-tunée est comparable à la
    représentation frozen (même pooling). ``head`` : ``nn.Linear(embed_dim, num_classes)``.
    """
    global _SSL_CLASSIFIER_CLASS
    if _SSL_CLASSIFIER_CLASS is not None:
        return _SSL_CLASSIFIER_CLASS
    import torch.nn as nn

    class SSLBackboneClassifier(nn.Module):
        def __init__(self, backbone, forward_fn, embed_dim, num_classes):
            super().__init__()
            self.backbone = backbone
            self._forward_fn = forward_fn
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            feats = self._forward_fn(self.backbone, x)
            return self.head(feats)

    _SSL_CLASSIFIER_CLASS = SSLBackboneClassifier
    return _SSL_CLASSIFIER_CLASS


def build_model(name: str, regime: str, num_classes: int, drop_path_rate: float = 0.1,
                lora=None, checkpoint: str | None = None):
    """Construit (model, param_groups) pour le fine-tuning.

    ``param_groups`` : dict ``{nom_groupe: [Parameters]}`` ; les LR sont appliqués par
    :func:`engine.build_optimizer` à partir de ``config.optim.lr`` (mêmes noms de groupes).
    ``regime='scratch'`` : init ALÉATOIRE (``pretrained=False``), 1 seul groupe 'all'
    (uniquement pour les ViT timm — pas de sens pour un backbone SSL pré-entraîné).
    ``regime='explora_like'`` : LoRA précoce + full-FT tardif (requiert ``lora`` = cfg.lora) ;
    supporté pour les ViT timm (qkv fusionné) ET les backbones ``SSL_FT_NAMES`` HuggingFace
    (projections Q/K/V séparées, ex. DINOv3) — cf. :func:`_explora_groups`.
    ``checkpoint`` : requis pour les backbones ``SSL_FT_NAMES`` de type SimDINOv2 (chemin du
    ``.pth`` teacher pré-entraîné — voir :func:`build_frozen_extractor`) ; ignoré sinon.
    """
    name = name.lower()
    _validate_regime(name, regime)  # peut lever AVANT tout import lourd

    if name in SSL_FT_NAMES:
        backbone, forward_fn, embed_dim, _norm_key = build_frozen_extractor(name, checkpoint)
        cls = _get_ssl_classifier_class()
        model = cls(backbone, forward_fn, embed_dim, num_classes)
        if regime == "frozen":
            _set_requires_grad(model, lambda n: "head" in n)
            return model, {"head": [p for n, p in model.named_parameters() if "head" in n]}
        if regime in ("explora_like", "lora"):
            return model, _explora_groups(model, lora)
        return model, _vit_groups(model, regime)

    import timm

    if name in CNN_NAMES:
        model = timm.create_model(_TIMM_ID[name], pretrained=True, num_classes=num_classes)
        return model, _resnet_groups(model, regime)

    # ViT (timm) — scratch = init aléatoire (borne inférieure), sinon poids ImageNet
    pretrained = regime != "scratch"
    model = timm.create_model(_TIMM_ID[name], pretrained=pretrained, num_classes=num_classes,
                              drop_path_rate=drop_path_rate)
    if regime == "scratch":
        _set_requires_grad(model, lambda n: True)
        return model, {"all": [p for _, p in model.named_parameters()]}
    if regime in ("explora_like", "lora"):
        return model, _explora_groups(model, lora)
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


def _cls_from_forward_features(model, x):
    """CLS token (index 0) depuis ``forward_features`` d'un ViT (SatMAE / ScaleMAE).

    ``forward_features`` renvoie les tokens ``(B, N, dim)`` — on prend le CLS ``[:, 0]``.
    Replis : dict (clé ``x_norm_clstoken`` / ``cls_token`` / ``pooler_output``), tuple/list,
    ou sortie déjà réduite ``(B, dim)``.
    """
    out = model.forward_features(x)
    if isinstance(out, dict):
        for k in ("x_norm_clstoken", "cls_token", "pooler_output"):
            if k in out:
                return out[k]
        out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        out = out[:, 0]
    return out


def _meanpool_from_forward_features(model, x):
    """Mean-pool des patch tokens (hors CLS) depuis ``forward_features`` — convention MAE.

    Pour les modèles MAE (SatMAE / ScaleMAE), le token CLS n'est PAS entraîné comme
    représentation (contrairement à DINO/DINOv2/v3) : il est quasi-constant et donne un
    espace effondré (anisotropie ~0.99, RankMe ~42 mesurés sur ScaleMAE). La recette
    linear-eval canonique des MAE est le global average pooling des patch tokens.
    ``forward_features`` renvoie ``(B, N, dim)`` avec le CLS en position 0 → on moyenne ``[:, 1:]``.
    Replis dict/tuple identiques à :func:`_cls_from_forward_features` ; sortie déjà réduite (B,dim) inchangée.
    """
    out = model.forward_features(x)
    if isinstance(out, dict):
        out = out.get("x_norm_patchtokens", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        out = out[:, 1:].mean(dim=1)
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
    if name == "dinov3_vits16_lvd":
        # HuggingFace (reproductibilité notebook legacy) : transformers.AutoModel + pooler_output
        # Réf. : facebook/dinov3-vits16-pretrain-lvd1689m, dim=384, 6 têtes, 12 blocs,
        # norm ImageNet. ViT-S/16 : ~22M params (~4× moins cher que ViT-B/16 par échantillon).
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "dinov3_vits16_lvd nécessite transformers>=4.56.0 : pip install transformers")
        m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        return m, _dinov3_hf_forward, 384, "imagenet"
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
    if name == "dinov3_vith16plus_lvd":
        # HuggingFace : transformers.AutoModel + pooler_output
        # Réf. : facebook/dinov3-vith16plus-pretrain-lvd1689m, dim=1280, 840M params,
        # ViT-H+/16, SwiGLU FFN, 20 têtes, RoPE, 4 register tokens, norm ImageNet
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "dinov3_vith16plus_lvd nécessite transformers>=4.56.0 : pip install transformers")
        m = AutoModel.from_pretrained("facebook/dinov3-vith16plus-pretrain-lvd1689m")
        return m, _dinov3_hf_forward, 1280, "imagenet"
    if name in ("simdinov2_vitb16", "simdinov2_vitl16",
                "simdinov2_vitb16_imagenet", "simdinov2_vitl16_imagenet"):
        # Deux pré-entraînements SimDINOv2, MÊME recette/arch (DINOv2-with-registers, 4 reg.,
        # patch16, block_chunks remappés par load_simdino_state_dict) → MÊME loader :
        #   - iNat21 Plantae (sslplant ilyassmoummad) → normalisation plantes `simdino_inat` ;
        #   - ImageNet-1k (officiel RobinWu218/SimDINO, teacher .pth) → normalisation `imagenet`.
        # Chaque encodeur figé reçoit les tuiles dans la distribution de son pré-entraînement
        # (norm native) — c'est ce qui rend la comparaison ImageNet vs iNat équitable.
        arch = "vitb" if "vitb16" in name else "vitl"
        dim = 768 if arch == "vitb" else 1024
        norm = "imagenet" if name.endswith("_imagenet") else "simdino_inat"
        if not checkpoint:
            raise ValueError(
                f"{name} : aucun checkpoint fourni. Renseigne le champ `checkpoint:` "
                f"(chemin du .pth teacher SimDINOv2) dans configs/frozen_{name}.yaml.")
        model = _load_simdinov2(arch, checkpoint)
        return model, _simdino_forward, dim, norm
    if name == "satmae_vitl16":
        # SatMAE ViT-L/16 (fMoW-RGB). Chargé dans un ViT-L/16 timm standard (num_classes=0)
        # par remap strict=False — SatMAE est bâti sur timm, la majorité des clés matchent.
        # ⚠ Utiliser le checkpoint fMoW-**RGB** (3 canaux), PAS fMoW-Sentinel (multispectral).
        if not checkpoint:
            raise ValueError(
                "satmae_vitl16 : aucun checkpoint fourni. Renseigne `checkpoint:` "
                "(chemin du .pth SatMAE fMoW-RGB ViT-L) dans configs/frozen_satmae.yaml.")
        m = _load_satmae(checkpoint)
        return m, _meanpool_from_forward_features, 1024, "imagenet"
    if name == "scalemae_vitl16":
        # ScaleMAE ViT-L/16 (fMoW-RGB) via TorchGeo (poids téléchargés automatiquement).
        # ⚠ Positional embeddings dépendants du GSD : entraîné satellite (~0.3-3 m/px), nos
        # tuiles UAV sont à ~2.5 mm/px — hors régime nominal (à garder en tête à l'analyse).
        try:
            from torchgeo.models import ScaleMAELarge16_Weights, scalemae_large_patch16
        except ImportError:
            raise ImportError("scalemae_vitl16 nécessite torchgeo : pip install torchgeo")
        m = scalemae_large_patch16(weights=ScaleMAELarge16_Weights.FMOW_RGB)
        return m, _meanpool_from_forward_features, 1024, "imagenet"
    if name in ("vitb16_arctic", "vitb16_fulft_arctic"):
        # ViT-B/16 fine-tuné Arctic-TVC (mhsa pour _arctic, full pour _fulft_arctic).
        # Backbone timm vit_base_patch16_224 num_classes=0 → CLS pooled (cohérent avec
        # l'entraînement, global_pool='token' par défaut). Checkpoint requis (poids sur Drive).
        if not checkpoint:
            raise ValueError(
                f"{name} : aucun checkpoint fourni. Renseigne `checkpoint:` (chemin du .pth "
                f"fine-tuné, p.ex. vitb16_mhsa_best.pth / vitb16_full_best.pth) dans configs/ft_*.yaml.")
        m = _load_finetuned_backbone("vit_base_patch16_224", checkpoint, 768)
        return m, _forward_direct, 768, "imagenet"
    if name == "resnet50_arctic":
        # ResNet-50 fine-tuné Arctic-TVC. Backbone timm resnet50 num_classes=0 → (B, 2048)
        # global-pooled. Checkpoint requis (poids sur Drive).
        if not checkpoint:
            raise ValueError(
                f"{name} : aucun checkpoint fourni. Renseigne `checkpoint:` (chemin du .pth "
                "fine-tuné, p.ex. resnet50_full_best.pth) dans configs/ft_resnet50_arctic.yaml.")
        m = _load_finetuned_backbone("resnet50", checkpoint, 2048)
        return m, _forward_direct, 2048, "imagenet"
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

    Réutilise l'arch officielle (``initialize_encoder`` : vit_base/vit_large DINOv2-with-
    registers, 4 registres, patch16, block_chunks=0) et le ``remap_key`` officiel (aplatit les
    blocs imbriqués ``blocks.N.M``→``blocks.N`` issus de block_chunks>0 à l'entraînement).
    On reproduit la logique de ``load_simdino_state_dict`` (clé ``teacher`` → suppression du
    préfixe ``backbone.`` → ``remap_key``) MAIS avec notre propre ``torch.load`` en
    ``map_location='cpu', weights_only=False`` : le loader du repo charge sans map_location, ce
    qui échoue sur une machine CPU-only pour un checkpoint sauvegardé sur CUDA (cas du .pth
    ImageNet officiel). On charge en ``strict=False`` et on lève si >10% des clés du modèle
    manquent (sécurité ; le repo, lui, utilise strict=True sur un dict déjà nettoyé).
    """
    import os
    import torch
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint SimDINOv2 introuvable : {checkpoint}")
    _ensure_sslplant_on_path()
    from get_model import initialize_encoder, remap_key  # simdinov2/eval/get_model.py

    encoder = initialize_encoder(arch=arch)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = raw["teacher"]
    teacher = {k.removeprefix("backbone."): v for k, v in teacher.items() if k.startswith("backbone.")}
    state_dict = {remap_key(k): v for k, v in teacher.items()}
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


def _load_satmae(checkpoint: str):
    """Charge un ViT-L/16 timm et y injecte les poids SatMAE (fMoW-RGB) en strict=False.

    SatMAE est bâti sur timm ViT : les clés (``cls_token``, ``pos_embed``, ``blocks.*``,
    ``norm.*``) matchent un ``vit_large_patch16_224`` timm. On charge ``num_classes=0`` (head
    Identity), on retire un éventuel wrapper ``model``/``state_dict``, un préfixe ``module.``
    et la tête de classif du checkpoint, puis on garde si <10 % des clés du modèle manquent
    (même garde-fou que SimDINOv2). Évite de dépendre du timm==0.3.2 épinglé par le repo SatMAE.
    """
    import os
    import timm
    import torch
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint SatMAE introuvable : {checkpoint}")
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    incompatible = model.load_state_dict(state, strict=False)
    n_model = len(model.state_dict())
    n_missing = len(incompatible.missing_keys)
    frac = n_missing / max(1, n_model)
    print(f"[satmae] vitl16: chargé {n_model - n_missing}/{n_model} clés "
          f"(manquantes={n_missing}, inattendues={len(incompatible.unexpected_keys)})")
    if frac > 0.10:
        raise RuntimeError(
            f"SatMAE vitl16 : {n_missing}/{n_model} clés manquantes ({frac:.0%} > 10%) — "
            f"checkpoint incompatible (clés inattendues={len(incompatible.unexpected_keys)}). "
            "Vérifie qu'il s'agit bien d'un ViT-L/16 fMoW-RGB (3 canaux).")
    return model.eval()


def _load_finetuned_backbone(arch: str, checkpoint: str, expected_dim: int):
    """Charge un backbone timm (``arch``) avec des poids fine-tunés Arctic-TVC, tête retirée.

    Sert aux 3 modèles fine-tunés (vitb16_arctic, vitb16_fulft_arctic, resnet50_arctic).
    On crée un timm ``num_classes=0`` (donc head/fc = Identity → la sortie est le backbone nu)
    et on y injecte les poids du checkpoint en ``strict=False`` avec le MÊME garde-fou que
    :func:`_load_satmae` (lève si >10 % des clés du modèle manquent).

    Le checkpoint vient de :func:`engine._save_ckpt` (poids sous ``model_state_dict``) ou d'un
    notebook (dict nu / ``model`` / ``state_dict``) — tous gérés. On retire les préfixes
    ``module.``/``backbone.`` et les clés de tête de classif (``head.*`` pour ViT, ``fc.*`` pour
    ResNet) qui n'existent pas dans le backbone ``num_classes=0``.
    """
    import os
    import timm
    import torch
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint fine-tuné introuvable : {checkpoint!r}")
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    if getattr(model, "num_features", expected_dim) != expected_dim:
        raise RuntimeError(
            f"{arch} : num_features={model.num_features} ≠ dim attendue {expected_dim}.")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict",
                         ckpt.get("model", ckpt.get("state_dict", ckpt)))
    else:
        state = ckpt
    # Régime explora_like : fusionne les adaptateurs LoRA dans qkv.weight (no-op sinon)
    # → l'extraction redevient agnostique au régime (ViT timm standard).
    state = merge_lora_state_dict(state)
    cleaned = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if nk.startswith("head.") or nk.startswith("fc."):
            continue  # tête de classif : absente du backbone num_classes=0
        cleaned[nk] = v
    incompatible = model.load_state_dict(cleaned, strict=False)
    n_model = len(model.state_dict())
    n_missing = len(incompatible.missing_keys)
    frac = n_missing / max(1, n_model)
    print(f"[finetuned] {arch}: chargé {n_model - n_missing}/{n_model} clés "
          f"(manquantes={n_missing}, inattendues={len(incompatible.unexpected_keys)})")
    if frac > 0.10:
        raise RuntimeError(
            f"{arch} fine-tuné : {n_missing}/{n_model} clés manquantes ({frac:.0%} > 10%) — "
            f"checkpoint incompatible (clés inattendues={len(incompatible.unexpected_keys)}). "
            "Vérifie l'architecture et le bon .pth.")
    return model.eval()


def load_finetuned_ssl_backbone(name: str, pretrain_checkpoint: str | None, ft_checkpoint: str):
    """Reconstruit un backbone SSL (``SSL_FT_NAMES``) et y injecte des poids fine-tunés.

    Réutilise :func:`build_frozen_extractor` pour l'architecture/``forward_fn``/normalisation
    (MÊME logique que l'extraction frozen — les poids pré-entraînés qu'il charge sont
    immédiatement écrasés par ceux du checkpoint fine-tuné ci-dessous, léger surcoût mais
    zéro divergence d'architecture possible). ``ft_checkpoint`` vient de
    :func:`engine._save_ckpt` (poids sous ``model_state_dict``, préfixe ``backbone.`` posé par
    :class:`SSLBackboneClassifier`) — on ne garde que ce sous-arbre (la tête ``head.*`` est
    ignorée : seule la représentation pré-tête sert à la sonde linéaire aval).
    Retourne ``(model, forward_fn, embed_dim, norm_key)`` — même signature que
    :func:`build_frozen_extractor`, prêt pour la même extraction (cache, sanity_check, etc.).
    """
    import os
    import torch
    if name not in SSL_FT_NAMES:
        raise ValueError(f"'{name}' n'est pas un backbone SSL fine-tunable ({sorted(SSL_FT_NAMES)}).")
    if not os.path.exists(ft_checkpoint):
        raise FileNotFoundError(f"checkpoint fine-tuné introuvable : {ft_checkpoint!r}")
    model, forward_fn, dim, norm_key = build_frozen_extractor(name, pretrain_checkpoint)
    ckpt = torch.load(ft_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = merge_lora_state_dict(state)  # fusionne si explora_like ; no-op sinon (full/mhsa)
    backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    if not backbone_state:
        raise RuntimeError(
            f"{name} : aucune clé 'backbone.*' dans {ft_checkpoint!r} — checkpoint inattendu "
            "(pas produit par SSLBackboneClassifier ?).")
    incompatible = model.load_state_dict(backbone_state, strict=False)
    n_model = len(model.state_dict())
    n_missing = len(incompatible.missing_keys)
    frac = n_missing / max(1, n_model)
    print(f"[finetuned-ssl] {name}: chargé {n_model - n_missing}/{n_model} clés "
          f"(manquantes={n_missing}, inattendues={len(incompatible.unexpected_keys)})")
    if frac > 0.10:
        raise RuntimeError(
            f"{name} fine-tuné : {n_missing}/{n_model} clés manquantes ({frac:.0%} > 10%) — "
            f"checkpoint incompatible (clés inattendues={len(incompatible.unexpected_keys)}).")
    return model.eval(), forward_fn, dim, norm_key


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
      - HuggingFace DINOv3 (``DINOv3ViTModel``, ``transformers.models.dinov3_vit``, vérifié en
        instantiant le modèle localement) : ``model.model.layer`` — ``DINOv3ViTModel.model``
        est un ``DINOv3ViTEncoder`` dont ``.layer`` est la ``ModuleList`` des blocs (PAS
        ``model.layer`` directement, contrairement à une supposition précédente non vérifiée).

    Si un modèle est passé déjà wrappé (ex. :class:`SSLBackboneClassifier` : ``model.backbone``
    est le vrai ``DINOv3ViTModel``), aucun des chemins ci-dessus ne matche directement ``model``
    → repli sur la plus longue ``nn.ModuleList`` trouvée par recherche récursive (``model.modules()``),
    qui retrouve la même ModuleList quel que soit le niveau d'imbrication. Le garde-fou de
    :func:`extract_layerwise` valide ensuite que ``len(blocks) == nb attendu`` (24 pour ViT-L).
    Lève ``ValueError`` pour un modèle sans blocs (ex. CNN).
    """
    for attr_path in (("blocks",), ("model", "layer"), ("layer",), ("layers",),
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