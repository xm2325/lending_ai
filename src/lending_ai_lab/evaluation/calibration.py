from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """One-dimensional logistic calibration fitted on a later validation cohort."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    @staticmethod
    def _logit(score: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, score: np.ndarray, target: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._logit(score), target)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(score))[:, 1]
