from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def recall_at_fraction(target: np.ndarray, score: np.ndarray, fraction: float = 0.10) -> float:
    n_selected = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.argsort(score)[-n_selected:]
    positives = target.sum()
    return float(target[selected].sum() / positives) if positives else float("nan")


def exposure_capture_at_fraction(
    target: np.ndarray,
    score: np.ndarray,
    exposure: np.ndarray,
    fraction: float = 0.10,
) -> float:
    n_selected = max(1, int(np.ceil(len(score) * fraction)))
    selected = np.argsort(score)[-n_selected:]
    total_loss_exposure = np.sum(exposure * target)
    return (
        float(np.sum(exposure[selected] * target[selected]) / total_loss_exposure)
        if total_loss_exposure > 0
        else float("nan")
    )


def expected_calibration_error(
    target: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(target[mask].mean()) - float(score[mask].mean()))
    return float(ece)


def classification_metrics(
    target: np.ndarray,
    score: np.ndarray,
    exposure: np.ndarray | None = None,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    auc = roc_auc_score(target, score)
    metrics = {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "average_precision": float(average_precision_score(target, score)),
        "brier": float(brier_score_loss(target, score)),
        "ece": expected_calibration_error(target, score),
        "recall_at_10pct": recall_at_fraction(target, score, top_fraction),
    }
    if exposure is not None:
        metrics["exposure_capture_at_10pct"] = exposure_capture_at_fraction(
            target, score, exposure, top_fraction
        )
    return metrics
