"""Confidence statistics from TypeSafe's public confidence documentation.

Choice: (p_max - 1/n) / (1 - 1/n)
Score: 1 - (probability-weighted distance from the peak) / (uniform MAD), floored at 0
Noul has no separate confidence in the API. |2p - 1| is the optional same-scale summary.
"""

from __future__ import annotations


def choice_confidence(probabilities: list[float]) -> float:
    if len(probabilities) < 2:
        raise ValueError("choice confidence needs at least two options")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    n = len(probabilities)
    peak = max(probabilities) / total
    return (peak - 1 / n) / (1 - 1 / n)


def score_confidence(probabilities: list[float]) -> float:
    if len(probabilities) < 2:
        raise ValueError("score confidence needs at least two levels")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive number")
    n = len(probabilities)
    normalized = [value / total for value in probabilities]
    peak = max(range(n), key=lambda index: normalized[index])
    spread = sum(prob * abs(index - peak) for index, prob in enumerate(normalized))
    even_spread = sum(abs(index - (n - 1) / 2) for index in range(n)) / n
    if even_spread == 0:
        return 1.0
    return max(0.0, 1 - spread / even_spread)


def noul_confidence(probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("noul probability must be between 0 and 1")
    return abs(2 * probability - 1)
