from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def _rows(frame: pd.DataFrame, columns: list[str]) -> str:
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(body)


def build_dashboard(artifacts: Path, output: Path) -> None:
    summary = json.loads((artifacts / "deep_summary.json").read_text(encoding="utf-8"))
    gate = json.loads((artifacts / "promotion_gate.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(artifacts / "deep_model_metrics.csv")
    paired = pd.read_csv(artifacts / "deep_paired_differences.csv")
    stress = pd.read_csv(artifacts / "deep_stress_tests.csv")
    columns = [
        "model",
        "roc_auc",
        "average_precision",
        "brier_score",
        "ece_10",
        "recall_at_10pct",
        "exposure_capture_at_10pct",
    ]
    paired_columns = ["metric", "improvement", "ci_low", "ci_high", "probability_improvement"]
    stress_columns = ["scenario", "mean_score_shift", "top_10pct_overlap", "roc_auc", "brier_score"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lending AI real-data review</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--ink:#eef5ff;--muted:#9db0c9;--line:#243650;--accent:#74c0fc;--good:#8ce99a;--warn:#ffd43b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:42px 20px 80px}}h1{{font-size:42px;margin:0 0 8px}}h2{{margin-top:36px}}p{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}
.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}}.value{{font-size:25px;font-weight:750;margin-top:5px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:block;overflow-x:auto}}
th,td{{padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{color:var(--accent)}}
.badge{{display:inline-block;padding:6px 10px;border-radius:999px;background:#203a2a;color:var(--good);font-weight:700}}.note{{border-left:4px solid var(--warn);padding:10px 14px;background:#241f0d}}
</style></head><body><main>
<div class="label">Real public-data decision review</div><h1>Lending AI Lab</h1>
<p>Shared-fold comparison of a strong tabular champion with LSTM, TCN and Transformer challengers. This page does not claim UK production validity.</p>
<div class="grid">
<div class="card"><div class="label">Champion</div><div class="value">{html.escape(summary['champion'])}</div></div>
<div class="card"><div class="label">Best deep challenger</div><div class="value">{html.escape(summary['best_deep_challenger'])}</div></div>
<div class="card"><div class="label">Technical gate</div><div class="value"><span class="badge">{html.escape(gate['technical_status'])}</span></div></div>
<div class="card"><div class="label">Customers</div><div class="value">{summary['n_customers']:,}</div></div>
</div>
<h2>Model scorecard</h2><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{_rows(metrics, columns)}</tbody></table>
<h2>Paired uncertainty</h2><p>Positive improvement means the challenger is better. Confidence intervals are customer-level paired bootstrap intervals.</p>
<table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in paired_columns)}</tr></thead><tbody>{_rows(paired, paired_columns)}</tbody></table>
<h2>Feed and stress sensitivity</h2><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in stress_columns)}</tr></thead><tbody>{_rows(stress, stress_columns)}</tbody></table>
<h2>Decision boundary</h2><div class="note">Research benchmark only. Production promotion requires point-in-time UK data, policy-selection analysis, outcome maturity, customer-impact review and independent validation.</div>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_dashboard(args.artifacts, args.output)


if __name__ == "__main__":
    main()
