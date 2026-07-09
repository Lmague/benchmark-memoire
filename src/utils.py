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
import re

import numpy as np

# --- Classes (ordre EXACT — ne JAMAIS réordonner : les labels CSV en dépendent) ---
CLASS_NAMES: list[str] = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS",
                          "PETF", "RHOL", "RUBC", "SEDG", "TUSS", "WILL"]
N_CLASSES: int = len(CLASS_NAMES)
RHOL_IDX: int = CLASS_NAMES.index("RHOL")              # 7 — absente du split test
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# --- Q5 datacurve (11 classes sans RHOL) ---
CLASS_NAMES_11: list[str] = [c for c in CLASS_NAMES if c != "RHOL"]
CLASS_TO_IDX_11: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES_11)}
LABEL_REMAP_12TO11: dict[int, int] = (
    {i: i for i in range(RHOL_IDX)} |
    {i: i - 1 for i in range(RHOL_IDX + 1, N_CLASSES)}
)

# --- Schéma 8 classes diagnostique (11cls moins ARCA/DRYI/RUBC) ---
# Les 3 classes retirées sont celles jamais évaluables de façon fiable (F1≈0 en
# probe/FT : ARCA, DRYI, RUBC). Identique à ``datacurve_one_run.LABELS_8CLS`` :
#   11-class : ALDE(0) ARCA(1) BIRC(2) DRYI(3) LICH(4) MOSS(5) PETF(6) RUBC(7) SEDG(8) TUSS(9) WILL(10)
#   8-class  : ALDE BIRC LICH MOSS PETF SEDG TUSS WILL  (labels 11-class [0,2,4,5,6,8,9,10])
_DROP_8CLS_NAMES: tuple[str, ...] = ("ARCA", "DRYI", "RUBC")
CLASS_NAMES_8: list[str] = [c for c in CLASS_NAMES_11 if c not in _DROP_8CLS_NAMES]

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


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fixe random/numpy/torch/cuda.

    ``deterministic=False`` (défaut, comportement historique) : ``cudnn.benchmark=True``
    (perf maximale, résultats non bit-pour-bit reproductibles entre runs/GPU).
    ``deterministic=True`` : ``cudnn.deterministic=True`` + ``benchmark=False`` (reproductible
    mais ~20 % plus lent ; ne PAS activer par défaut en production).
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
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


# --- Schémas de labels par SOURCE d'embeddings -----------------------------
# Deux conventions coexistent dans le corpus élargi :
#   "12cls" : embeddings CANONIQUES (dinov3_*, simdino_*, satmae_*, *_arctic, ...).
#             Labels 0–11 (RHOL=7 présent au train ; absent du val/test mais la
#             numérotation reste 0–11).
#   "11cls" : runs SOTA (``vitb16_{regime}_frac{XXX}_seed{N}``). RHOL retiré À
#             L'EXTRACTION → labels 0–10 (``schema: 11cls_no_rhol``), indices > 7
#             déjà comblés (RUBC=7, SEDG=8, TUSS=9, WILL=10).
# Preuve (valeurs uniques des test_labels) documentée dans le rapport de tâche.
_SOTA_KEY_RE = re.compile(r"^vitb16_(full|mhsa|explora|scratch)_frac\d{3}_seed\d+$")

# Noms de classes restants par passe (indépendant de la source).
_PASS_NAMES: dict[str, list[str]] = {
    "with_rhol": list(CLASS_NAMES),      # 12
    "without_rhol": list(CLASS_NAMES_11),  # 11
    "8cls": list(CLASS_NAMES_8),          # 8
}
PROBE_PASS_TAGS: tuple[str, ...] = ("with_rhol", "without_rhol", "8cls")


def is_sota_key(key: str) -> bool:
    """True si ``key`` est un run SOTA (``vitb16_{regime}_frac{XXX}_seed{N}``)."""
    return bool(_SOTA_KEY_RE.match(key))


def sota_regime(key: str) -> str:
    """Régime (full|mhsa|explora|scratch) d'une clé SOTA (ValueError sinon)."""
    m = _SOTA_KEY_RE.match(key)
    if not m:
        raise ValueError(f"clé SOTA invalide : {key!r}")
    return m.group(1)


def source_schema(key: str) -> str:
    """Schéma de labels bruts d'un modèle : '11cls' (SOTA no-rhol) sinon '12cls'."""
    return "11cls" if is_sota_key(key) else "12cls"


def pass_drops(tag: str, source_schema: str = "12cls") -> list[int] | None:
    """Labels à retirer (ordre DÉCROISSANT, pour cascade ``drop_class``) pour la
    passe ``tag`` depuis une source ``source_schema`` ('12cls' | '11cls').

    Retourne ``None`` si la passe n'est PAS applicable à cette source (à sauter) :
    c'est le cas de ``with_rhol`` sur une source SOTA 11cls (RHOL déjà absent).
    L'ordre décroissant est OBLIGATOIRE : ``drop_class`` décale les labels > drop de
    -1, donc retirer du plus grand indice au plus petit évite tout désalignement
    (vérifié par test unitaire ; l'ordre croissant retire les mauvaises classes).
    """
    if source_schema == "12cls":
        table: dict[str, list[int] | None] = {
            "with_rhol": [],                       # tel quel (12cls)
            "without_rhol": [CLASS_TO_IDX["RHOL"]],  # [7]
            "8cls": sorted((CLASS_TO_IDX[c] for c in ("RHOL", *_DROP_8CLS_NAMES)),
                           reverse=True),           # [8, 7, 3, 1]
        }
    elif source_schema == "11cls":
        table = {
            "with_rhol": None,                     # RHOL déjà absent → non applicable
            "without_rhol": [],                    # déjà 11cls, rien à retirer
            "8cls": sorted((CLASS_TO_IDX_11[c] for c in _DROP_8CLS_NAMES),
                           reverse=True),           # [7, 3, 1]
        }
    else:
        raise ValueError(f"source_schema inconnu : {source_schema!r} (12cls | 11cls)")
    if tag not in table:
        raise ValueError(f"passe inconnue : {tag!r} (dispo: {sorted(table)})")
    return table[tag]


def probe_passes() -> list[tuple[str, list[str]]]:
    """Les 3 passes d'évaluation : ``[(tag, noms_de_classes_restants), ...]``.

    Les labels à retirer par passe dépendent de la SOURCE de chaque modèle et sont
    obtenus via :func:`pass_drops` (par-modèle dans ``probe.py``), car les modèles
    canoniques (12cls) et SOTA (11cls) ne partagent pas le même référentiel d'indices.
    """
    return [(t, list(_PASS_NAMES[t])) for t in PROBE_PASS_TAGS]


def rhol_passes(source_schema: str = "12cls") -> list[tuple[str, list[int] | None, list[str]]]:
    """Passes pour une source HOMOGÈNE : ``[(tag, drops, noms), ...]``.

    - ``with_rhol``    : 12 classes (source 12cls) / non applicable (source 11cls).
    - ``without_rhol`` : 11 classes (RHOL retiré si 12cls, tel quel si 11cls).
    - ``8cls``         : 8 classes (11cls moins ARCA/DRYI/RUBC).

    ``drops`` est une liste DÉCROISSANTE de labels à retirer en cascade (``[]`` = rien
    à retirer), ou ``None`` si la passe n'est pas applicable à la source. Conservé pour
    ``analyze.py`` et les usages homogènes ; ``probe.py`` mélange les sources et utilise
    :func:`probe_passes` + :func:`pass_drops` par modèle.
    """
    return [(t, pass_drops(t, source_schema), names) for t, names in probe_passes()]


def maybe_mount_drive(env: str) -> None:
    """Monte Google Drive sur Colab uniquement.

    ``google.colab`` est importé ICI, jamais au niveau module, pour préserver
    l'import local du package (``python -c "import src"``).
    """
    if env != "colab":
        return
    if os.path.isdir("/content/drive/MyDrive"):
        return  # déjà monté (cellule notebook) — éviter drive.mount() en subprocess
    from google.colab import drive  # import local volontaire (Colab seulement)
    drive.mount("/content/drive")


# ------------------------------------------------------------------ Linear probe
# Helper unique pour instancier la régression logistique selon la MÉTHODOLOGIE
# CANONIQUE du benchmark. À utiliser partout au lieu d'instancier
# ``LogisticRegression(...)`` à la main, pour garantir la cohérence des kwargs.
#
# Notes sklearn :
#   - ``solver="lbfgs"`` ⇒ multi-classes intrinsèquement multinomial depuis
#     longtemps (l'arg ``multi_class`` était ignoré depuis 0.22 puis supprimé
#     en 1.8 — donc on le passe seulement quand sklearn le supporte encore,
#     sinon le simple fait de fixer ``solver="lbfgs"`` suffit à garantir le
#     comportement multinomial).
#   - ``n_jobs=-1`` : utilise tous les cœurs pour le fit (cohérent avec
#     ``joblib.Parallel`` utilisé en amont dans ``src.probe.linear_probe``).
_CANONICAL_LR_SUPPORTS_MULTICLASS = True
try:
    from sklearn.linear_model import LogisticRegression as _LR
    import inspect as _inspect
    _CANONICAL_LR_SUPPORTS_MULTICLASS = (
        "multi_class" in _inspect.signature(_LR).parameters
    )
except Exception:  # pragma: no cover - sklearn toujours dispo, filet de sécurité
    _CANONICAL_LR_SUPPORTS_MULTICLASS = False


def make_canonical_lr(C: float, max_iter: int = 2000, random_state: int = 42) -> "object":
    """Régression logistique canonique : lbfgs + multinomial + random_state=42.

    - Passe ``multi_class="multinomial"`` quand sklearn le supporte (<1.8).
    - Sinon, ne le passe pas : ``lbfgs`` est intrinsèquement multinomial.
    - ``random_state=42`` partout pour la reproductibilité bit-pour-bit.
    - ``n_jobs`` : volontairement omis (ignoré depuis sklearn 1.8, warning sinon).
    """
    from sklearn.linear_model import LogisticRegression
    kwargs = dict(
        C=float(C),
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    if _CANONICAL_LR_SUPPORTS_MULTICLASS:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def stratified_subsample(labels: "np.ndarray", fraction: float, seed: int) -> "np.ndarray":
    """Sous-échantillonnage stratifié : maintient les proportions de classes.

    Garantit au moins 1 exemple par classe présente.
    ``fraction=1.0`` retourne tous les indices (pas de copie).
    """
    rng = np.random.RandomState(seed)
    if fraction >= 1.0:
        return np.arange(len(labels), dtype=np.int64)
    indices_out: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n_cls = max(1, int(round(len(cls_idx) * fraction)))
        n_cls = min(n_cls, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_cls, replace=False)
        indices_out.extend(chosen.tolist())
    return np.array(sorted(indices_out), dtype=np.int64)
