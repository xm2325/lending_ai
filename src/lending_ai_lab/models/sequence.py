from __future__ import annotations

import torch
from torch import nn


class LSTMFusion(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.ReLU(), nn.LayerNorm(16))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        seq_repr = hidden[-1]
        static_repr = self.static_net(static)
        return self.head(torch.cat([seq_repr, static_repr], dim=1)).squeeze(1)


class GRUFusion(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.LayerNorm(16))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        seq_repr = hidden[-1]
        static_repr = self.static_net(static)
        return self.head(torch.cat([seq_repr, static_repr], dim=1)).squeeze(1)


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.norm((x + y).transpose(1, 2)).transpose(1, 2)


class TCNFusion(nn.Module):
    """Small temporal convolutional network for short account histories."""

    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(sequence_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _CausalResidualBlock(hidden_dim, dilation=1, dropout=dropout),
            _CausalResidualBlock(hidden_dim, dilation=2, dropout=dropout),
        )
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.LayerNorm(16))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        x = self.blocks(self.input_projection(sequence.transpose(1, 2))).transpose(1, 2)
        positions = torch.arange(sequence.shape[1], device=sequence.device).unsqueeze(0)
        valid = (positions < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        static_repr = self.static_net(static)
        return self.head(torch.cat([pooled, static_repr], dim=1)).squeeze(1)


class TransformerFusion(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        static_dim: int,
        max_length: int = 6,
        model_dim: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.input_projection = nn.Linear(sequence_dim, model_dim)
        self.position = nn.Parameter(torch.zeros(1, max_length, model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.static_net = nn.Sequential(nn.Linear(static_dim, 16), nn.GELU(), nn.LayerNorm(16))
        self.head = nn.Sequential(
            nn.Linear(model_dim + 16, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = sequence.shape
        x = self.input_projection(sequence) + self.position[:, :seq_len]
        positions = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch, -1)
        padding_mask = positions >= lengths.unsqueeze(1)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        static_repr = self.static_net(static)
        return self.head(torch.cat([pooled, static_repr], dim=1)).squeeze(1)
