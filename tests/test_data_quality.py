from pathlib import Path

from lending_ai_lab.evaluation.data_quality import inspect_uci_source


def test_sample_source_has_no_hard_quality_failures():
    report = inspect_uci_source(Path("data/sample/UCI_Credit_Card_sample.csv"))
    assert {"check", "severity", "status", "observed", "expectation"}.issubset(report.columns)
    assert not (report.status == "fail").any()
    assert (report.check == "row_count").any()
