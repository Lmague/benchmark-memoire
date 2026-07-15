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

def build_class_weights_effective_num(labels: Iterable[int], n_classes: int, beta: float = 0.999):
    """Poids Cui et al. 2019 : w_k = (1-beta) / (1 - beta**n_k), normalisés à N_CLASSES."""
    import torch
    counts = torch.tensor(class_counts(labels, n_classes), dtype=torch.float)
    eff_num = 1.0 - torch.pow(beta, counts)
    w = (1.0 - beta) / eff_num
    w = w / w.sum() * n_classes
    return w

def build_criterion(labels, n_classes, device=None, weighting: str = "sqrt", beta: float = 0.999):
    """``nn.CrossEntropyLoss`` pondérée, schéma choisi par ``weighting`` (``sqrt`` | ``effective_num``)."""
    import torch.nn as nn
    w = (build_class_weights(labels, n_classes) if weighting == "sqrt"
         else build_class_weights_effective_num(labels, n_classes, beta))
    if device is not None:
        w = w.to(device)
    return nn.CrossEntropyLoss(weight=w)
    
def test_build_criterion_weighting_switch():
    import torch
    labels = [0]*100 + [1]*10 + [2]*1
    crit_sqrt = build_criterion(labels, 3, weighting="sqrt")
    crit_eff = build_criterion(labels, 3, weighting="effective_num", beta=0.999)
    assert not torch.allclose(crit_sqrt.weight, crit_eff.weight), \
        "les deux schémas de pondération donnent le même poids — bug de branchement"
