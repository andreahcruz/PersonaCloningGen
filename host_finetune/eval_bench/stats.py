"""Interval estimates for paired architecture comparisons.

These intervals describe the recorded questions. They do not create a single
winner score.
"""

from __future__ import annotations

import math
import random


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> list[float] | None:
    """95 percent Wilson interval for a rate. Empty input stays unset."""
    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes must lie inside the total")
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z ** 2 / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def paired_mean_difference(
    left: list[float],
    right: list[float],
    *,
    seed: int = 298,
    draws: int = 1000,
) -> dict:
    """Bootstrap the mean of left minus right on aligned questions."""
    if len(left) != len(right) or not left:
        raise ValueError("paired comparison needs equal non-empty lists")
    if draws < 100:
        raise ValueError("use at least 100 bootstrap draws")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    observed = sum(differences) / len(differences)
    generator = random.Random(seed)
    samples = []
    width = len(differences)
    for _ in range(draws):
        draw = [differences[generator.randrange(width)] for _ in range(width)]
        samples.append(sum(draw) / width)
    samples.sort()
    low_index = int(0.025 * (draws - 1))
    high_index = int(0.975 * (draws - 1))
    return {
        "n": width,
        "mean_difference": observed,
        "ci95": [samples[low_index], samples[high_index]],
        "seed": seed,
        "draws": draws,
        "direction": "left_minus_right",
    }
