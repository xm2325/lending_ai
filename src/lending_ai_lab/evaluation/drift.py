from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_table(
    reference_sequence: np.ndarray,
    current_sequence: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(feature_names):
        rows.append(
            {
                "feature": name,
                "psi": population_stability_index(
                    reference_sequence[..., index].ravel(),
                    current_sequence[..., index].ravel(),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
