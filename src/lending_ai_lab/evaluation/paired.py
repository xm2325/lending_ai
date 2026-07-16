from __future__ import annotations

from collections.abc import Callable

import numpy as np


Metric = Callable[[np.ndarray, np.ndarray], float]


def paired_bootstrap_difference(
    y: np.ndarray,
    challenger: np.ndarray,
    champion: np.ndarray,
    metric: Metric,
    *,
    higher_is_better: bool = True,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Paired customer-level bootstrap for challenger minus champion performance."""
    y = np.asarray(y)
    challenger = np.asarray(challenger)
    champion = np.asarray(champion)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size < 2:
            continue
        difference = metric(y[index], challenger[index]) - metric(y[index], champion[index])
        if not higher_is_better:
            difference = -difference
        differences.append(float(difference))
    values = np.asarray(differences)
    point = metric(y, challenger) - metric(y, champion)
    if not higher_is_better:
        point = -point
    return {
        "improvement": float(point),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_improvement": float((values > 0).mean()),
    }
