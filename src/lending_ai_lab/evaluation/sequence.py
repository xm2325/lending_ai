from __future__ import annotations

import numpy as np
import pandas as pd

from lending_ai_lab.evaluation.metrics import classification_metrics
from lending_ai_lab.training import TorchTrainingResult, predict_torch


def prefix_performance(
    model_name: str,
    result: TorchTrainingResult,
    sequence: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    exposure: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for length in range(1, sequence.shape[1] + 1):
        truncated = sequence.copy()
        truncated[:, length:] = 0.0
        lengths = np.full(len(sequence), length, dtype=np.int64)
        score = predict_torch(result, truncated, static, lengths=lengths)
        metrics = classification_metrics(target, score, exposure)
        rows.append({"model": model_name, "months_observed": length, **metrics})
    return pd.DataFrame(rows)


def month_occlusion_importance(
    result: TorchTrainingResult,
    sequence: np.ndarray,
    static: np.ndarray,
) -> pd.DataFrame:
    base = predict_torch(result, sequence, static)
    rows = []
    for month in range(sequence.shape[1]):
        occluded = sequence.copy()
        occluded[:, month, :] = 0.0
        score = predict_torch(result, occluded, static)
        rows.append(
            {
                "month_index": month + 1,
                "mean_absolute_score_change": float(np.mean(np.abs(base - score))),
            }
        )
    return pd.DataFrame(rows)
