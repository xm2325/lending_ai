from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def latest_month_features(sequence: np.ndarray, static: np.ndarray) -> np.ndarray:
    return np.concatenate([static, sequence[:, -1, :]], axis=1)


def flattened_history_features(sequence: np.ndarray, static: np.ndarray) -> np.ndarray:
    return np.concatenate([static, sequence.reshape(len(sequence), -1)], axis=1)


def fit_logistic(sequence: np.ndarray, static: np.ndarray, target: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(latest_month_features(sequence, static), target)
    return model


def fit_tree(sequence: np.ndarray, static: np.ndarray, target: np.ndarray) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.06,
        max_iter=180,
        l2_regularization=0.5,
        random_state=42,
    )
    model.fit(flattened_history_features(sequence, static), target)
    return model
