"""Loss : CrossEntropy pondérée par fréquence inverse en 1/sqrt(n), normalisée à N_CLASSES.

Source de vérité = notebooks. La pondération est portée UNIQUEMENT par la loss, jamais
couplée à un WeightedRandomSampler (cf. mémoire projet : pas de stacking sampler+loss).
torch importé paresseusement.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def class_counts(labels: Iterable[int], n_classes: int) -> list[int]:
    """Effectifs par classe (minimum 1 pour éviter une division par zéro)."""
    c = Counter(int(x) for x in labels)
    return [c.get(i, 1) for i in range(n_classes)]


def build_class_weights(labels: Iterable[int], n_classes: int):
    """Poids ``w_k = 1/sqrt(n_k)`` normalisés tels que ``sum(w) == n_classes``.

    Reproduit exactement le calcul des notebooks :
        class_weights = 1/sqrt(counts) ; class_weights = class_weights / sum * N_CLASSES
    """
    import torch
    counts = torch.tensor(class_counts(labels, n_classes), dtype=torch.float)
    w = 1.0 / torch.sqrt(counts)
    w = w / w.sum() * n_classes
    return w


def build_criterion(labels: Iterable[int], n_classes: int, device=None):
    """``nn.CrossEntropyLoss`` pondérée par :func:`build_class_weights`."""
    import torch.nn as nn
    w = build_class_weights(labels, n_classes)
    if device is not None:
        w = w.to(device)
    return nn.CrossEntropyLoss(weight=w)
