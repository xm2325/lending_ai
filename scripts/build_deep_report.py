from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}")
    return display.to_markdown(index=False)


def build_report(artifacts: Path, output: Path) -> None:
    summary = json.loads((artifacts / "deep_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((artifacts / "promotion_gate.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(artifacts / "deep_model_metrics.csv")
    paired = pd.read_csv(artifacts / "deep_paired_differences.csv")
    order = pd.read_csv(artifacts / "deep_order_sensitivity.csv")
    stress = pd.read_csv(artifacts / "deep_stress_tests.csv")
    quality = pd.read_csv(artifacts / "data_quality_checks.csv")

    champion = metrics.loc[metrics.model == summary["champion"]].iloc[0]
    challenger = metrics.loc[metrics.model == summary["best_deep_challenger"]].iloc[0]
    lines = [
        "# Deep sequence benchmark on real public credit data",
        "",
        "This report is generated from shared customer-level folds on the official UCI dataset. "
        "It is a research benchmark, not a production underwriting validation.",
        "",
        "## Decision first",
        "",
        f"- Tabular champion: **{summary['champion']}** (ROC AUC {champion.roc_auc:.4f})",
        f"- Best deep challenger: **{summary['best_deep_challenger']}** "
        f"(ROC AUC {challenger.roc_auc:.4f})",
        f"- Technical gate: **{gate['technical_status']}**",
        f"- Overall best model: **{summary['best_model_overall']}**",
        "- Calibration uses fold-local validation customers; test customers are not used to fit the calibrator.",
        "- Production use remains blocked because the public source has no point-in-time UK policy data.",
        "",
        "## Shared-fold comparison",
        "",
        _table(metrics),
        "",
        "## Paired customer-level uncertainty",
        "",
        "Positive values mean the deep challenger improves on the tabular champion. For Brier score, "
        "the sign is reversed so positive still means better calibration error.",
        "",
        _table(paired),
        "",
        "## Does the neural model use month order?",
        "",
        _table(order),
        "",
        "## Input and feed sensitivity",
        "",
        "These are controlled perturbations, not macroeconomic forecasts. They test how model scores and "
        "the top-10% review population respond when important feeds are missing or shifted.",
        "",
        _table(stress),
        "",
        "## Automated data checks",
        "",
        _table(quality),
        "",
        "## Release blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in gate["production_blockers"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.artifacts, args.output)


if __name__ == "__main__":
    main()
