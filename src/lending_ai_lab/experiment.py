from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from lending_ai_lab.data.synthetic import generate_synthetic_credit_data
from lending_ai_lab.evaluation.metrics import classification_metrics
from lending_ai_lab.models.baselines import fit_logistic, fit_tree, flattened_history_features, latest_month_features
from lending_ai_lab.models.sequence import LSTMFusion, TransformerFusion
from lending_ai_lab.reporting import build_dashboard
from lending_ai_lab.training import predict_torch, train_torch_model


def run_demo(output_dir="artifacts", n_samples: int = 12_000, epochs: int = 8, seed: int = 42):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_credit_data(n_samples=n_samples, seed=seed)
    train = data.cohort < 8
    validation = (data.cohort >= 8) & (data.cohort < 10)
    test = data.cohort >= 10

    logistic = fit_logistic(data.sequence[train], data.static[train], data.target[train])
    tree = fit_tree(data.sequence[train], data.static[train], data.target[train])
    lstm = train_torch_model(
        LSTMFusion(5, 3), data.sequence[train], data.static[train], data.target[train],
        data.sequence[validation], data.static[validation], data.target[validation],
        epochs=epochs, seed=seed
    )
    transformer = train_torch_model(
        TransformerFusion(5, 3), data.sequence[train], data.static[train], data.target[train],
        data.sequence[validation], data.static[validation], data.target[validation],
        epochs=epochs, seed=seed + 1
    )
    scores = {
        "Logistic latest month": logistic.predict_proba(latest_month_features(data.sequence[test], data.static[test]))[:, 1],
        "Tree full history": tree.predict_proba(flattened_history_features(data.sequence[test], data.static[test]))[:, 1],
        "LSTM fusion": predict_torch(lstm, data.sequence[test], data.static[test]),
        "Transformer fusion": predict_torch(transformer, data.sequence[test], data.static[test]),
    }
    rows = [
        {"model": name, **classification_metrics(data.target[test], score, data.exposure[test])}
        for name, score in scores.items()
    ]
    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    summary = {
        "best_model": str(metrics.iloc[0].model),
        "best_roc_auc": float(metrics.iloc[0].roc_auc),
        "synthetic_data": True,
        "n_samples": int(n_samples),
        "seed": int(seed),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_dashboard(metrics, summary, Path("site/index.html"))
    return summary
