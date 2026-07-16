import numpy as np

from lending_ai_lab.evaluation.paired import paired_bootstrap_difference
from lending_ai_lab.evaluation.reliability import (
    PlattCalibrator,
    calibration_table,
    expected_calibration_error,
)


def test_calibration_outputs_are_finite():
    target = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    probability = np.array([0.05, 0.20, 0.35, 0.55, 0.70, 0.80, 0.95, 0.25])
    calibrated = PlattCalibrator().fit(probability, target).predict(probability)
    assert calibrated.shape == probability.shape
    assert np.isfinite(calibrated).all()
    assert 0 <= expected_calibration_error(target, calibrated) <= 1
    assert calibration_table(target, calibrated).n.sum() == len(target)


def test_paired_bootstrap_detects_better_predictions():
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 20)
    champion = np.tile([0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8], 20)
    challenger = np.tile([0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95], 20)
    from sklearn.metrics import brier_score_loss

    result = paired_bootstrap_difference(
        target,
        challenger,
        champion,
        brier_score_loss,
        higher_is_better=False,
        n_bootstrap=100,
    )
    assert result["improvement"] > 0
    assert result["probability_improvement"] > 0.95
