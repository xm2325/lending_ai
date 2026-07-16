from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def subgroup_metrics(
    target: np.ndarray,
    score: np.ndarray,
    groups: pd.Series,
    minimum_group_size: int = 100,
) -> pd.DataFrame:
    rows = []
    for group in sorted(groups.astype(str).unique()):
        mask = groups.astype(str).to_numpy() == group
        if mask.sum() < minimum_group_size or len(np.unique(target[mask])) < 2:
            continue
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "default_rate": float(target[mask].mean()),
                "roc_auc": float(roc_auc_score(target[mask], score[mask])),
                "brier": float(brier_score_loss(target[mask], score[mask])),
                "mean_score": float(score[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
