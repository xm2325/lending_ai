from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from lending_ai_lab.data.uci_default import load_uci_default
from lending_ai_lab.evaluation.data_quality import inspect_uci_source
from lending_ai_lab.evaluation.paired import paired_bootstrap_difference
from lending_ai_lab.evaluation.reliability import PlattCalibrator, binary_metrics, calibration_table
from lending_ai_lab.models.baselines import flattened_history_features
from lending_ai_lab.models.sequence import LSTMFusion, TCNFusion, TransformerFusion
from lending_ai_lab.training import predict_torch, train_torch_model


def _capture_at_fraction(
    target: np.ndarray,
    probability: np.ndarray,
    exposure: np.ndarray,
    fraction: float = 0.10,
) -> tuple[float, float]:
    n_selected = max(1, int(np.ceil(len(target) * fraction)))
    selected = np.argsort(-probability)[:n_selected]
    positives = target == 1
    recall = float(target[selected].sum() / max(target.sum(), 1))
    weighted = exposure * positives
    exposure_capture = float(weighted[selected].sum() / max(weighted.sum(), 1e-12))
    return recall, exposure_capture


def _metric_row(
    name: str,
    target: np.ndarray,
    probability: np.ndarray,
    exposure: np.ndarray,
    training_seconds: float,
) -> dict[str, float | str]:
    metrics = binary_metrics(target, probability)
    recall, exposure_capture = _capture_at_fraction(target, probability, exposure)
    return {
        "model": name,
        **metrics,
        "recall_at_10pct": recall,
        "exposure_capture_at_10pct": exposure_capture,
        "training_seconds": float(training_seconds),
    }


def _model_factories(sequence_dim: int, static_dim: int) -> dict[str, Callable[[], torch.nn.Module]]:
    return {
        "LSTM": lambda: LSTMFusion(sequence_dim, static_dim, hidden_dim=24, dropout=0.10),
        "TCN": lambda: TCNFusion(sequence_dim, static_dim, hidden_dim=24, dropout=0.10),
        "Transformer": lambda: TransformerFusion(
            sequence_dim,
            static_dim,
            model_dim=24,
            nhead=4,
            num_layers=1,
            dropout=0.10,
        ),
    }


def _permuted_sequence(sequence: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(sequence[:, permutation, :])


def _stress_variants(sequence: np.ndarray, feature_names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    full_lengths = np.full(len(sequence), sequence.shape[1], dtype=np.int64)
    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "baseline": (sequence, full_lengths),
        "reversed_history": (np.ascontiguousarray(sequence[:, ::-1, :]), full_lengths),
    }

    missing_latest = sequence.copy()
    missing_latest[:, -1, :] = 0.0
    variants["latest_month_unavailable"] = (
        missing_latest,
        np.full(len(sequence), sequence.shape[1] - 1, dtype=np.int64),
    )

    if "payment_ratio" in feature_names:
        delayed_payment = sequence.copy()
        delayed_payment[:, -1, feature_names.index("payment_ratio")] = 0.0
        variants["latest_payment_feed_missing"] = (delayed_payment, full_lengths)

    if "delinquency" in feature_names:
        missing_delinquency = sequence.copy()
        missing_delinquency[:, :, feature_names.index("delinquency")] = 0.0
        variants["delinquency_feed_missing"] = (missing_delinquency, full_lengths)

    if "utilisation" in feature_names:
        shock = sequence.copy()
        util_index = feature_names.index("utilisation")
        shock[:, -1, util_index] = np.clip(shock[:, -1, util_index] * 1.20, -1.0, 3.0)
        if "balance_change" in feature_names:
            change_index = feature_names.index("balance_change")
            shock[:, -1, change_index] = shock[:, -1, util_index] - shock[:, -2, util_index]
        variants["latest_utilisation_plus_20pct"] = (shock, full_lengths)

    return variants


def _paired_exposure_bootstrap(
    target: np.ndarray,
    challenger: np.ndarray,
    champion: np.ndarray,
    exposure: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(target), len(target))
        if np.unique(target[index]).size < 2:
            continue
        challenger_capture = _capture_at_fraction(target[index], challenger[index], exposure[index])[1]
        champion_capture = _capture_at_fraction(target[index], champion[index], exposure[index])[1]
        values.append(challenger_capture - champion_capture)
    point = _capture_at_fraction(target, challenger, exposure)[1] - _capture_at_fraction(
        target, champion, exposure
    )[1]
    array = np.asarray(values)
    return {
        "improvement": float(point),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "probability_improvement": float((array > 0).mean()),
    }


def _promotion_gate(
    best_deep: str,
    paired: pd.DataFrame,
    order_sensitivity: pd.DataFrame,
    data_quality: pd.DataFrame,
) -> dict[str, object]:
    by_metric = paired.set_index("metric")
    auc_noninferior = float(by_metric.loc["roc_auc", "ci_low"]) > -0.005
    calibration_noninferior = float(by_metric.loc["brier_score", "ci_low"]) > -0.002
    operational_signal = (
        float(by_metric.loc["average_precision", "probability_improvement"]) >= 0.90
        or float(by_metric.loc["exposure_capture_at_10pct", "improvement"]) >= 0.01
    )
    order = order_sensitivity.set_index("control")
    order_signal = (
        "reversed_history" in order.index
        and float(order.loc["baseline", "roc_auc"] - order.loc["reversed_history", "roc_auc"]) >= 0.001
    )
    hard_failures = int((data_quality["status"] == "fail").sum())

    technical_status = (
        "shadow_candidate"
        if auc_noninferior and calibration_noninferior and operational_signal and hard_failures == 0
        else "retain_tabular_champion"
    )
    return {
        "challenger": best_deep,
        "champion": "Histogram gradient boosting",
        "technical_status": technical_status,
        "research_release_scope": "public benchmark only",
        "auc_noninferiority_gate": auc_noninferior,
        "calibration_noninferiority_gate": calibration_noninferior,
        "operational_signal_gate": operational_signal,
        "sequence_order_signal": order_signal,
        "hard_data_quality_failures": hard_failures,
        "production_blockers": [
            "no point-in-time application timestamps",
            "no UK population or policy data",
            "no rejected-applicant outcomes",
            "no loss-given-default or treatment-effect evidence",
            "no independent model validation",
        ],
    }


def run_deep_uci_benchmark(
    path: str | Path,
    output_dir: str | Path = "artifacts_deep",
    *,
    folds: int = 3,
    epochs: int = 8,
    seed: int = 42,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    data = load_uci_default(path)
    target = data.target
    data_quality = inspect_uci_source(path)
    data_quality.to_csv(output / "data_quality_checks.csv", index=False)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    names = ["Histogram gradient boosting", "LSTM", "TCN", "Transformer"]
    predictions = {name: np.zeros(len(target), dtype=float) for name in names}
    fold_rows: list[dict[str, object]] = []
    perturbation_predictions: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in names if name != "Histogram gradient boosting"
    }
    total_training_seconds = {name: 0.0 for name in names}

    for fold, (outer_train, test_index) in enumerate(splitter.split(data.static, target), start=1):
        train_index, validation_index = train_test_split(
            outer_train,
            test_size=0.15,
            stratify=target[outer_train],
            random_state=seed + fold,
        )

        x = flattened_history_features(data.sequence, data.static)
        champion = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed + fold,
        )
        started = perf_counter()
        champion.fit(x[train_index], target[train_index])
        fold_training_seconds = perf_counter() - started
        total_training_seconds["Histogram gradient boosting"] += fold_training_seconds
        validation_probability = champion.predict_proba(x[validation_index])[:, 1]
        champion_calibrator = PlattCalibrator().fit(validation_probability, target[validation_index])
        calibrated = champion_calibrator.predict(champion.predict_proba(x[test_index])[:, 1])
        predictions["Histogram gradient boosting"][test_index] = calibrated
        fold_rows.append(
            {
                "fold": fold,
                **_metric_row(
                    "Histogram gradient boosting",
                    target[test_index],
                    calibrated,
                    data.exposure[test_index],
                    fold_training_seconds,
                ),
            }
        )

        factories = _model_factories(data.sequence.shape[-1], data.static.shape[-1])
        permutation = np.random.default_rng(seed + fold).permutation(data.sequence.shape[1])
        for model_offset, (name, factory) in enumerate(factories.items(), start=1):
            model_seed = seed + fold * 100 + model_offset
            torch.manual_seed(model_seed)
            result = train_torch_model(
                factory(),
                data.sequence[train_index],
                data.static[train_index],
                target[train_index],
                data.sequence[validation_index],
                data.static[validation_index],
                target[validation_index],
                epochs=epochs,
                batch_size=1024,
                learning_rate=2e-3,
                seed=model_seed,
                device="cpu",
            )
            total_training_seconds[name] += result.training_seconds
            validation_probability = predict_torch(
                result,
                data.sequence[validation_index],
                data.static[validation_index],
                batch_size=2048,
                device="cpu",
            )
            calibrator = PlattCalibrator().fit(validation_probability, target[validation_index])
            calibrated = calibrator.predict(
                predict_torch(
                    result,
                    data.sequence[test_index],
                    data.static[test_index],
                    batch_size=2048,
                    device="cpu",
                )
            )
            predictions[name][test_index] = calibrated
            fold_rows.append(
                {
                    "fold": fold,
                    **_metric_row(
                        name,
                        target[test_index],
                        calibrated,
                        data.exposure[test_index],
                        result.training_seconds,
                    ),
                }
            )

            variants = _stress_variants(data.sequence[test_index], data.sequence_feature_names)
            variants["shared_random_permutation"] = (
                _permuted_sequence(data.sequence[test_index], permutation),
                np.full(len(test_index), data.sequence.shape[1], dtype=np.int64),
            )
            for control, (variant_sequence, lengths) in variants.items():
                calibrated_variant = calibrator.predict(
                    predict_torch(
                        result,
                        variant_sequence,
                        data.static[test_index],
                        lengths=lengths,
                        batch_size=2048,
                        device="cpu",
                    )
                )
                perturbation_predictions[name].setdefault(control, np.zeros(len(target), dtype=float))[
                    test_index
                ] = calibrated_variant

    metrics = pd.DataFrame(
        [
            _metric_row(name, target, probability, data.exposure, total_training_seconds[name])
            for name, probability in predictions.items()
        ]
    ).sort_values("roc_auc", ascending=False)
    metrics.to_csv(output / "deep_model_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "deep_fold_metrics.csv", index=False)

    calibration_rows: list[pd.DataFrame] = []
    for name, probability in predictions.items():
        table = calibration_table(target, probability, n_bins=10)
        table.insert(0, "model", name)
        calibration_rows.append(table)
    pd.concat(calibration_rows, ignore_index=True).to_csv(output / "deep_calibration_bins.csv", index=False)

    best_deep = str(metrics[metrics["model"] != "Histogram gradient boosting"].iloc[0]["model"])
    champion_probability = predictions["Histogram gradient boosting"]
    challenger_probability = predictions[best_deep]
    paired_rows = []
    for metric_name, metric, higher_is_better in [
        ("roc_auc", roc_auc_score, True),
        ("average_precision", average_precision_score, True),
        ("brier_score", brier_score_loss, False),
    ]:
        row = paired_bootstrap_difference(
            target,
            challenger_probability,
            champion_probability,
            metric,
            higher_is_better=higher_is_better,
            n_bootstrap=500,
            seed=seed,
        )
        paired_rows.append({"metric": metric_name, **row})
    paired_rows.append(
        {
            "metric": "exposure_capture_at_10pct",
            **_paired_exposure_bootstrap(
                target,
                challenger_probability,
                champion_probability,
                data.exposure,
                seed=seed,
            ),
        }
    )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "deep_paired_differences.csv", index=False)

    order_rows = []
    stress_rows = []
    base_probability = perturbation_predictions[best_deep]["baseline"]
    selection_size = int(np.ceil(0.10 * len(target)))
    base_top = set(np.argsort(-base_probability)[:selection_size])
    for control, probability in perturbation_predictions[best_deep].items():
        row = _metric_row(best_deep, target, probability, data.exposure, 0.0)
        order_rows.append(
            {
                "control": control,
                **{key: value for key, value in row.items() if key not in {"model", "training_seconds"}},
            }
        )
        current_top = set(np.argsort(-probability)[:selection_size])
        stress_rows.append(
            {
                "scenario": control,
                "mean_score": float(probability.mean()),
                "mean_score_shift": float(probability.mean() - base_probability.mean()),
                "top_10pct_overlap": float(len(base_top & current_top) / max(len(base_top), 1)),
                "roc_auc": float(roc_auc_score(target, probability)),
                "brier_score": float(brier_score_loss(target, probability)),
            }
        )
    order_sensitivity = pd.DataFrame(order_rows)
    order_sensitivity.to_csv(output / "deep_order_sensitivity.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "deep_stress_tests.csv", index=False)

    gate = _promotion_gate(best_deep, paired, order_sensitivity, data_quality)
    (output / "promotion_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    summary: dict[str, object] = {
        "dataset": "UCI Default of Credit Card Clients",
        "n_customers": int(len(target)),
        "folds": folds,
        "epochs": epochs,
        "champion": "Histogram gradient boosting",
        "best_deep_challenger": best_deep,
        "best_model_overall": str(metrics.iloc[0]["model"]),
        "technical_status": gate["technical_status"],
        "validation": "Shared customer-level stratified folds; not temporal validation",
        "calibration": "Fold-local Platt scaling fitted only on inner validation customers",
        "production_status": "blocked by public-data limitations",
    }
    (output / "deep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
