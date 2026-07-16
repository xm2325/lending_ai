from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CreditSequenceDataset(Dataset):
    def __init__(self, sequence, static, target, training: bool = False, seed: int = 42) -> None:
        self.sequence = torch.as_tensor(sequence, dtype=torch.float32)
        self.static = torch.as_tensor(static, dtype=torch.float32)
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        seq = self.sequence[index].clone()
        length = seq.shape[0]
        if self.training and self.rng.random() < 0.25:
            length = int(self.rng.integers(max(2, seq.shape[0] // 2), seq.shape[0] + 1))
            seq[length:] = 0.0
        return seq, self.static[index], self.target[index], length


@dataclass
class TorchTrainingResult:
    model: nn.Module
    sequence_scaler: StandardScaler
    static_scaler: StandardScaler
    best_validation_loss: float
    training_seconds: float


def fit_scalers(train_sequence: np.ndarray, train_static: np.ndarray):
    seq_scaler = StandardScaler().fit(train_sequence.reshape(-1, train_sequence.shape[-1]))
    static_scaler = StandardScaler().fit(train_static)
    return seq_scaler, static_scaler


def apply_scalers(sequence, static, sequence_scaler, static_scaler):
    seq = sequence_scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
    sta = static_scaler.transform(static)
    return seq.astype(np.float32), sta.astype(np.float32)


def train_torch_model(
    model,
    train_sequence,
    train_static,
    train_target,
    validation_sequence,
    validation_static,
    validation_target,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    seed: int = 42,
    device: str | None = None,
) -> TorchTrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    seq_scaler, static_scaler = fit_scalers(train_sequence, train_static)
    tr_seq, tr_sta = apply_scalers(train_sequence, train_static, seq_scaler, static_scaler)
    va_seq, va_sta = apply_scalers(validation_sequence, validation_static, seq_scaler, static_scaler)
    train_loader = DataLoader(
        CreditSequenceDataset(tr_seq, tr_sta, train_target, True, seed),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(CreditSequenceDataset(va_seq, va_sta, validation_target), batch_size=batch_size)
    positive = float(train_target.sum())
    negative = float(len(train_target) - positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(positive, 1.0), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state, best_loss, stale = None, float("inf"), 0
    started = perf_counter()
    for _ in range(epochs):
        model.train()
        for sequence, static, target, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence.to(device), static.to(device), lengths.to(device))
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for sequence, static, target, lengths in val_loader:
                losses.append(float(loss_fn(
                    model(sequence.to(device), static.to(device), lengths.to(device)), target.to(device)
                ).cpu()))
        current = float(np.mean(losses))
        if current < best_loss - 1e-4:
            best_loss = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TorchTrainingResult(model, seq_scaler, static_scaler, best_loss, perf_counter() - started)


def predict_torch(result, sequence, static, *, lengths=None, batch_size: int = 512, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seq, sta = apply_scalers(sequence, static, result.sequence_scaler, result.static_scaler)
    if lengths is None:
        lengths = np.full(len(seq), seq.shape[1], dtype=np.int64)
    result.model.to(device).eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(seq), batch_size):
            end = start + batch_size
            logits = result.model(
                torch.as_tensor(seq[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(sta[start:end], dtype=torch.float32, device=device),
                torch.as_tensor(lengths[start:end], dtype=torch.long, device=device),
            )
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)
