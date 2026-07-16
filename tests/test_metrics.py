import numpy as np

from lending_ai_lab.evaluation.metrics import classification_metrics, recall_at_fraction


def test_recall_at_fraction_and_metric_ranges():
    target = np.array([0, 1, 0, 1, 1, 0])
    score = np.array([0.1, 0.8, 0.2, 0.9, 0.7, 0.3])
    assert recall_at_fraction(target, score, 0.5) == 1.0
    metrics = classification_metrics(target, score)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["brier"] <= 1
