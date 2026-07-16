from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


EPS = 1e-6


def safe_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    frame = calibration_table(y, p, n_bins=n_bins)
    return float((frame["weight"] * np.abs(frame["mean_prediction"] - frame["event_rate"])).sum())


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = bins == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "n": int(mask.sum()),
                "weight": float(mask.mean()),
                "mean_prediction": float(p[mask].mean()),
                "event_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CalibrationParameters:
    intercept: float
    slope: float


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> CalibrationParameters:
    x = safe_logit(p).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(x, np.asarray(y, dtype=int))
    return CalibrationParameters(intercept=float(model.intercept_[0]), slope=float(model.coef_[0, 0]))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "PlattCalibrator":
        self.model.fit(safe_logit(probability).reshape(-1, 1), np.asarray(target, dtype=int))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(safe_logit(probability).reshape(-1, 1))[:, 1]


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    parameters = calibration_intercept_slope(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
        "calibration_intercept": parameters.intercept,
        "calibration_slope": parameters.slope,
    }
