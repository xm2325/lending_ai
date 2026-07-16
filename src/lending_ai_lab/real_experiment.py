from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lending_ai_lab.data.uci_default import load_uci_default
from lending_ai_lab.models.baselines import flattened_history_features, latest_month_features


def _bootstrap_ci(y, p, metric, seed: int = 42, n: int = 400):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(float(metric(y[idx], p[idx])))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _oof_predict(estimator, x, y, seed: int):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probability = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        fitted = clone(estimator).fit(x[train_idx], y[train_idx])
        probability[test_idx] = fitted.predict_proba(x[test_idx])[:, 1]
    return probability


def _capture_at_fraction(y, p, exposure, fraction: float = 0.10):
    n_selected = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-p)[:n_selected]
    recall = float(y[selected].sum() / y.sum())
    bad_exposure = float((exposure * y).sum())
    exposure_capture = float((exposure[selected] * y[selected]).sum() / bad_exposure)
    return recall, exposure_capture


def run_uci_benchmark(path, output_dir="artifacts_real", seed: int = 42):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load_uci_default(path)
    y = data.target
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model_specs = {
        "Logistic latest month": (latest_month_features(data.sequence, data.static), logistic),
        "Logistic full history": (flattened_history_features(data.sequence, data.static), logistic),
        "Histogram gradient boosting": (
            flattened_history_features(data.sequence, data.static),
            HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed
            ),
        ),
    }
    rows, predictions = [], {}
    for name, (x, estimator) in model_specs.items():
        p = _oof_predict(estimator, x, y, seed)
        predictions[name] = p
        auc = roc_auc_score(y, p)
        lo, hi = _bootstrap_ci(y, p, roc_auc_score, seed)
        recall, exposure_capture = _capture_at_fraction(y, p, data.exposure)
        rows.append({
            "model": name,
            "roc_auc": auc,
            "roc_auc_ci_low": lo,
            "roc_auc_ci_high": hi,
            "average_precision": average_precision_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "recall_at_10pct": recall,
            "exposure_capture_at_10pct": exposure_capture,
        })
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(out / "real_model_metrics.csv", index=False)
    best = str(metrics.iloc[0].model)
    p = predictions[best]

    subgroup_rows = []
    for dimension in ["sex", "age_band"]:
        for group, idx_values in data.audit.groupby(dimension).groups.items():
            idx = np.asarray(list(idx_values), dtype=int)
            if len(idx) < 30 or len(np.unique(y[idx])) < 2:
                continue
            recall, exposure_capture = _capture_at_fraction(y[idx], p[idx], data.exposure[idx])
            subgroup_rows.append({
                "dimension": dimension, "group": str(group), "n": len(idx),
                "default_rate": float(y[idx].mean()),
                "roc_auc": roc_auc_score(y[idx], p[idx]),
                "brier_score": brier_score_loss(y[idx], p[idx]),
                "mean_score": float(p[idx].mean()),
                "recall_at_10pct": recall,
                "exposure_capture_at_10pct": exposure_capture,
            })
    pd.DataFrame(subgroup_rows).to_csv(out / "real_subgroup_metrics.csv", index=False)

    ablation_rows = []
    for months in range(1, 7):
        truncated = data.sequence.copy()
        truncated[:, :-months, :] = 0.0
        pred = _oof_predict(logistic, flattened_history_features(truncated, data.static), y, seed)
        ablation_rows.append({
            "months_available": months,
            "roc_auc": roc_auc_score(y, pred),
            "average_precision": average_precision_score(y, pred),
            "brier_score": brier_score_loss(y, pred),
        })
    pd.DataFrame(ablation_rows).to_csv(out / "real_history_ablation.csv", index=False)

    rng = np.random.default_rng(seed)
    controls = {
        "chronological": data.sequence,
        "reversed": data.sequence[:, ::-1, :],
        "shared_random_permutation": data.sequence[:, rng.permutation(data.sequence.shape[1]), :],
    }
    control_rows = []
    for label, sequence in controls.items():
        pred = _oof_predict(logistic, flattened_history_features(sequence, data.static), y, seed)
        control_rows.append({
            "control": label,
            "roc_auc": roc_auc_score(y, pred),
            "average_precision": average_precision_score(y, pred),
            "brier_score": brier_score_loss(y, pred),
        })
    pd.DataFrame(control_rows).to_csv(out / "real_sequence_controls.csv", index=False)

    summary = {
        "dataset": "UCI Default of Credit Card Clients",
        "n_customers": int(len(y)),
        "default_rate": float(y.mean()),
        "best_model": best,
        "best_roc_auc": float(metrics.iloc[0].roc_auc),
        "validation": "Five-fold stratified customer-level cross-validation; not temporal validation",
        "release_decision": "Research benchmark only; no production promotion without point-in-time UK data",
        "real_data": True,
    }
    (out / "real_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
