"""Arctic-TVC vegetation classification benchmark — package reproductible.

Importer :mod:`src` est volontairement léger : seul :mod:`src.utils` (numpy) est
chargé ici.  Les dépendances lourdes / optionnelles (torch, torchvision, timm,
scikit-learn, pandas, seaborn) sont importées paresseusement dans les sous-modules
qui en ont besoin, et ``google.colab`` n'est jamais importé au niveau module
(uniquement dans :func:`src.utils.maybe_mount_drive`).
"""
from __future__ import annotations

from .utils import CLASS_NAMES, CLASS_TO_IDX, N_CLASSES, RHOL_IDX

__all__ = ["CLASS_NAMES", "CLASS_TO_IDX", "N_CLASSES", "RHOL_IDX"]
__version__ = "0.1.0"
