from __future__ import annotations

import torch
from torch import nn


class LSTMFusion(nn.Module):
    def __init__(self, sequence_dim: int, static_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(sequence_dim, hidden_dim, batch_first=True)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU(), nn.LayerNorm(16))
        self.head = nn.Sequential(nn.Linear(hidden_dim + 16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat([hidden[-1], self.static_net(static)], dim=1)).squeeze(1)


class TransformerFusion(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        max_length: int = 6,
        model_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(sequence_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, max_length, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.LayerNorm(16))
        self.head = nn.Sequential(nn.Linear(model_dim + 16, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = sequence.shape
        x = self.input_projection(sequence) + self.position[:, :seq_len]
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch, -1)
        padding_mask = positions >= lengths.unsqueeze(1)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([pooled, self.static_net(static)], dim=1)).squeeze(1)
