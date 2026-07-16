from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def build_dashboard(metrics: pd.DataFrame, summary: dict, output_path: Path) -> None:
    rows = "".join(
        f"<tr><td>{escape(str(row.model))}</td><td>{row.roc_auc:.4f}</td>"
        f"<td>{row.average_precision:.4f}</td><td>{row.brier:.4f}</td>"
        f"<td>{row.recall_at_10pct:.2%}</td></tr>"
        for _, row in metrics.iterrows()
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Lending AI Lab</title>
<style>body{{font-family:system-ui;max-width:1050px;margin:auto;padding:40px;background:#f5f7fa;color:#152238}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}.card{{background:white;padding:24px;border-radius:12px;margin:18px 0}}</style></head><body>
<h1>Lending AI Lab</h1><div class='card'><h2>Decision evidence</h2><p>Best synthetic smoke-test model: <b>{escape(str(summary['best_model']))}</b>. These results are not production lending evidence.</p></div>
<div class='card'><h2>Model comparison</h2><table><thead><tr><th>Model</th><th>ROC AUC</th><th>Average precision</th><th>Brier</th><th>Recall at 10%</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Research demonstration • no automated lending decision</p></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
