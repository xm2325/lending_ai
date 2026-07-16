import torch

from lending_ai_lab.models.sequence import GRUFusion, LSTMFusion, TCNFusion, TransformerFusion


def test_sequence_models_return_one_logit_per_customer():
    sequence = torch.randn(8, 6, 5)
    static = torch.randn(8, 3)
    lengths = torch.tensor([6, 6, 5, 4, 3, 6, 2, 1])
    for model in [LSTMFusion(5, 3), GRUFusion(5, 3), TCNFusion(5, 3), TransformerFusion(5, 3)]:
        output = model(sequence, static, lengths)
        assert output.shape == (8,)
        assert torch.isfinite(output).all()
