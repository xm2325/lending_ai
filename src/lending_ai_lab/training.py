from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class CreditSequenceDataset(Dataset):
    def __init__(
        self,
        sequence: np.ndarray,
        static: np.ndarray,
        target: np.ndarray,
        training: bool = False,
        seed: int = 42,
    ) -> None:
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


def fit_scalers(
    train_sequence: np.ndarray, train_static: np.ndarray
) -> tuple[StandardScaler, StandardScaler]:
    sequence_scaler = StandardScaler().fit(train_sequence.reshape(-1, train_sequence.shape[-1]))
    static_scaler = StandardScaler().fit(train_static)
    return sequence_scaler, static_scaler


def apply_scalers(
    sequence: np.ndarray,
    static: np.ndarray,
    sequence_scaler: StandardScaler,
    static_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    seq = sequence_scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
    sta = static_scaler.transform(static)
    return seq.astype(np.float32), sta.astype(np.float32)


def train_torch_model(
    model: nn.Module,
    train_sequence: np.ndarray,
    train_static: np.ndarray,
    train_target: np.ndarray,
    validation_sequence: np.ndarray,
    validation_static: np.ndarray,
    validation_target: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str | None = None,
) -> TorchTrainingResult:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sequence_scaler, static_scaler = fit_scalers(train_sequence, train_static)
    tr_seq, tr_sta = apply_scalers(train_sequence, train_static, sequence_scaler, static_scaler)
    va_seq, va_sta = apply_scalers(validation_sequence, validation_static, sequence_scaler, static_scaler)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CreditSequenceDataset(tr_seq, tr_sta, train_target, training=True, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        CreditSequenceDataset(va_seq, va_sta, validation_target),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    positive = float(train_target.sum())
    negative = float(len(train_target) - positive)
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = None
    best_loss = float("inf")
    patience = 3
    stale = 0
    started = perf_counter()
    for _ in range(epochs):
        model.train()
        for sequence, static, target, lengths in train_loader:
            sequence = sequence.to(device)
            static = static.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence, static, lengths)
            loss = loss_fn(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for sequence, static, target, lengths in validation_loader:
                sequence = sequence.to(device)
                static = static.to(device)
                target = target.to(device)
                lengths = lengths.to(device)
                losses.append(float(loss_fn(model(sequence, static, lengths), target).cpu()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TorchTrainingResult(
        model=model,
        sequence_scaler=sequence_scaler,
        static_scaler=static_scaler,
        best_validation_loss=best_loss,
        training_seconds=perf_counter() - started,
    )


def predict_torch(
    result: TorchTrainingResult,
    sequence: np.ndarray,
    static: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
    batch_size: int = 512,
    device: str | None = None,
) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seq, sta = apply_scalers(
        sequence, static, result.sequence_scaler, result.static_scaler
    )
    if lengths is None:
        lengths = np.full(len(seq), seq.shape[1], dtype=np.int64)
    result.model.to(device).eval()
    output: list[np.ndarray] = []
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
