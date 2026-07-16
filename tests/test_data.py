import numpy as np

from lending_ai_lab.data.synthetic import generate_synthetic_credit_data


def test_synthetic_shapes_and_audit_separation():
    data = generate_synthetic_credit_data(n_samples=200, seed=7)
    assert data.sequence.shape == (200, 6, 5)
    assert data.static.shape == (200, 3)
    assert set(np.unique(data.target)).issubset({0, 1})
    assert "sex" in data.audit.columns
    assert "sex" not in data.static_feature_names
