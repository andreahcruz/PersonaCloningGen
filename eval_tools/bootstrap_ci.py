"""Paired differences with a confidence interval and a sign-flip test.

Kevin's comparison file reports only the mean and median difference between two
runs and says "significance: not computed; the evaluator has no paired test". This
adds the missing part.

Two choices that matter:

* The unit of resampling is the topic, not the draft. The four mediums written for
  one gold topic are not independent, so a plain draft level bootstrap would make
  the interval too narrow. Pass ``clusters`` (the topic id of each difference).
* Everything is deterministic for a fixed ``seed``.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from typing import Hashable, Sequence

MIN_PAIRS = 5


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation quantile on an already sorted list."""
    if not sorted_values:
        raise ValueError("no values")
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _group(values: Sequence[float], clusters: Sequence[Hashable] | None) -> list[tuple[float, int]]:
    """Return (sum, count) per cluster. Without clusters every value is its own cluster."""
    if clusters is None:
        return [(float(v), 1) for v in values]
    if len(clusters) != len(values):
        raise ValueError("values and clusters must have the same length")
    sums: dict[Hashable, float] = defaultdict(float)
    counts: dict[Hashable, int] = defaultdict(int)
    for value, cluster in zip(values, clusters):
        sums[cluster] += float(value)
        counts[cluster] += 1
    return [(sums[c], counts[c]) for c in sums]


def bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling whole clusters."""
    groups = _group(values, clusters)
    if not groups:
        raise ValueError("no values")
    rng = random.Random(seed)
    k = len(groups)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        count = 0
        for _ in range(k):
            s, c = groups[rng.randrange(k)]
            total += s
            count += c
        means.append(total / count)
    means.sort()
    alpha = (1 - ci) / 2
    return _quantile(means, alpha), _quantile(means, 1 - alpha)


def sign_flip_p(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    n_perm: int = 10000,
    seed: int = 0,
    exact_up_to: int = 14,
) -> float:
    """Two sided p value that the true mean difference is zero.

    Under the null each topic's differences are equally likely to have either sign,
    so we flip the sign of whole clusters. With up to ``exact_up_to`` clusters every
    sign pattern is tried (exact). With more, random patterns are used.
    """
    groups = _group(values, clusters)
    total_count = sum(c for _, c in groups)
    if not groups or total_count == 0:
        raise ValueError("no values")
    observed = abs(sum(s for s, _ in groups))
    if observed < 1e-12:
        return 1.0
    sums = [s for s, _ in groups]
    k = len(sums)
    tol = 1e-12
    if k <= exact_up_to:
        hits = 0
        n_patterns = 0
        for signs in itertools.product((1.0, -1.0), repeat=k):
            n_patterns += 1
            if abs(sum(sg * s for sg, s in zip(signs, sums))) >= observed - tol:
                hits += 1
        return hits / n_patterns
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        if abs(sum(s if rng.random() < 0.5 else -s for s in sums)) >= observed - tol:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def summarize_paired(
    diffs: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    n_boot: int = 10000,
    n_perm: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
    min_pairs: int = MIN_PAIRS,
) -> dict:
    """Mean difference (B minus A), interval, p value and effect size for paired data."""
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    out: dict = {
        "n": n,
        "n_clusters": len(set(clusters)) if clusters is not None else n,
        "mean_diff": None,
        "median_diff": None,
        "ci_low": None,
        "ci_high": None,
        "ci_level": ci,
        "p_value": None,
        "effect_size_dz": None,
        "share_zero_diff": None,
        "note": None,
    }
    if n == 0:
        out["note"] = "no paired drafts"
        return out
    out["mean_diff"] = round(statistics.fmean(diffs), 6)
    out["median_diff"] = round(statistics.median(diffs), 6)
    out["share_zero_diff"] = round(sum(1 for d in diffs if d == 0) / n, 4)
    spread = statistics.stdev(diffs) if n > 1 else 0.0
    out["effect_size_dz"] = round(statistics.fmean(diffs) / spread, 4) if spread > 0 else None
    if n < min_pairs or out["n_clusters"] < min_pairs:
        out["note"] = f"too few pairs for an interval (need at least {min_pairs})"
        return out
    low, high = bootstrap_ci(diffs, clusters, n_boot=n_boot, ci=ci, seed=seed)
    out["ci_low"] = round(low, 6)
    out["ci_high"] = round(high, 6)
    out["p_value"] = round(sign_flip_p(diffs, clusters, n_perm=n_perm, seed=seed), 6)
    return out


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    """Holm step down correction for a family of tests. None stays None."""
    items = sorted(((k, p) for k, p in p_values.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float | None] = {k: None for k in p_values}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = round(running, 6)
    return adjusted
