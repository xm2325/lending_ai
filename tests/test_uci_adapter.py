from pathlib import Path

from lending_ai_lab.data.uci_default import load_uci_default


def test_uci_csv_adapter_shapes_and_order():
    data = load_uci_default(Path("data/sample/UCI_Credit_Card_sample.csv"))
    assert data.sequence.shape == (20, 6, 5)
    assert data.static.shape == (20, 3)
    assert data.target.shape == (20,)
    assert data.audit.shape[0] == 20
    assert data.sequence_feature_names[-1] == "zero_bill"
    assert 0 < data.target.mean() < 1
