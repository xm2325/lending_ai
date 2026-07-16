from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEQUENCE_FEATURES = [
    "utilisation",
    "payment_ratio",
    "delinquency",
    "balance_change",
    "cash_share",
]
STATIC_FEATURES = ["log_credit_limit", "tenure_years", "active_accounts"]
AUDIT_FEATURES = ["sex", "age_band"]


@dataclass
class DatasetBundle:
    sequence: np.ndarray
    static: np.ndarray
    target: np.ndarray
    exposure: np.ndarray
    cohort: np.ndarray
    audit: pd.DataFrame
    sequence_feature_names: list[str]
    static_feature_names: list[str]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))


def generate_synthetic_credit_data(
    n_samples: int = 12_000,
    sequence_length: int = 6,
    seed: int = 42,
) -> DatasetBundle:
    """Create a realistic but non-customer credit sequence for reproducible testing.

    The generator creates temporal stress, utilisation, payment and delinquency patterns.
    Protected fields are returned only for post-model audit and are never model inputs.
    """
    rng = np.random.default_rng(seed)
    cohort = rng.integers(0, 12, size=n_samples)

    credit_limit = np.exp(rng.normal(np.log(8_000), 0.65, size=n_samples)).clip(500, 50_000)
    tenure = rng.gamma(shape=2.2, scale=2.0, size=n_samples).clip(0.1, 20)
    active_accounts = rng.poisson(2.2, size=n_samples).clip(1, 9)
    static = np.column_stack([np.log1p(credit_limit), tenure, active_accounts]).astype(np.float32)

    sex = rng.choice(["female", "male"], size=n_samples, p=[0.53, 0.47])
    age = rng.normal(41, 12, size=n_samples).clip(18, 79)
    age_band = pd.cut(age, bins=[17, 29, 44, 59, 100], labels=["18-29", "30-44", "45-59", "60+"])

    base_stress = rng.normal(0, 0.75, size=n_samples)
    drift = 0.055 * np.maximum(cohort - 7, 0)
    trend = rng.normal(0.02, 0.10, size=n_samples) + drift

    utilisation = np.zeros((n_samples, sequence_length), dtype=np.float32)
    payment_ratio = np.zeros_like(utilisation)
    delinquency = np.zeros_like(utilisation)
    balance_change = np.zeros_like(utilisation)
    cash_share = np.zeros_like(utilisation)

    util_prev = _sigmoid(-0.45 + 0.55 * base_stress + rng.normal(0, 0.55, n_samples))
    for t in range(sequence_length):
        stress_t = base_stress + trend * t + rng.normal(0, 0.35, n_samples)
        util_t = np.clip(0.70 * util_prev + 0.30 * _sigmoid(stress_t), 0.01, 1.35)
        pay_t = np.clip(_sigmoid(0.95 - 1.15 * stress_t + rng.normal(0, 0.55, n_samples)), 0.01, 1.5)
        delinquency_prob = _sigmoid(-2.15 + 1.35 * stress_t + 0.9 * (util_t > 0.9))
        delinquency_t = rng.binomial(3, delinquency_prob, n_samples) / 3.0
        change_t = np.clip(util_t - util_prev + rng.normal(0, 0.06, n_samples), -0.5, 0.8)
        cash_t = np.clip(_sigmoid(-1.65 + 0.9 * stress_t + rng.normal(0, 0.5, n_samples)), 0, 1)

        utilisation[:, t] = util_t
        payment_ratio[:, t] = pay_t
        delinquency[:, t] = delinquency_t
        balance_change[:, t] = change_t
        cash_share[:, t] = cash_t
        util_prev = util_t

    sequence = np.stack(
        [utilisation, payment_ratio, delinquency, balance_change, cash_share], axis=-1
    ).astype(np.float32)

    util_slope = utilisation[:, -1] - utilisation[:, 0]
    pay_slope = payment_ratio[:, -1] - payment_ratio[:, 0]
    volatility = utilisation.std(axis=1)
    recent_delinq = delinquency[:, -2:].mean(axis=1)
    latent = (
        -3.15
        + 1.55 * utilisation[:, -1]
        + 1.30 * util_slope
        - 0.95 * payment_ratio[:, -1]
        - 0.55 * pay_slope
        + 1.35 * recent_delinq
        + 1.20 * volatility
        + 0.20 * np.log1p(active_accounts)
        - 0.10 * tenure
        + 0.10 * np.maximum(cohort - 9, 0)
        + rng.normal(0, 0.45, n_samples)
    )
    probability = _sigmoid(latent)
    target = rng.binomial(1, probability, n_samples).astype(np.int64)
    exposure = (credit_limit * (0.25 + 0.75 * utilisation[:, -1])).astype(np.float32)

    audit = pd.DataFrame({"sex": sex, "age_band": age_band.astype(str), "age": age})
    return DatasetBundle(
        sequence=sequence,
        static=static,
        target=target,
        exposure=exposure,
        cohort=cohort,
        audit=audit,
        sequence_feature_names=SEQUENCE_FEATURES.copy(),
        static_feature_names=STATIC_FEATURES.copy(),
    )
