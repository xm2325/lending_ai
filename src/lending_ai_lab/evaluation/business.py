from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_value_curve(
    target: np.ndarray,
    score: np.ndarray,
    exposure: np.ndarray,
    lgd: float = 0.55,
    intervention_cost_rate: float = 0.015,
) -> pd.DataFrame:
    """Estimate a transparent decision value curve, not a production P&L claim."""
    rows = []
    for threshold in np.linspace(0.05, 0.80, 31):
        selected = score >= threshold
        avoided_loss = np.sum(exposure[selected] * target[selected] * lgd * 0.25)
        intervention_cost = np.sum(exposure[selected] * intervention_cost_rate)
        rows.append(
            {
                "threshold": threshold,
                "selected_rate": float(selected.mean()),
                "avoided_loss_proxy": float(avoided_loss),
                "intervention_cost_proxy": float(intervention_cost),
                "net_value_proxy": float(avoided_loss - intervention_cost),
            }
        )
    return pd.DataFrame(rows)
