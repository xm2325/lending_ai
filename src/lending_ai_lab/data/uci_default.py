from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .synthetic import AUDIT_FEATURES, DatasetBundle

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)


def _read_frame(path_or_url: str | Path) -> pd.DataFrame:
    source = str(path_or_url)
    if source.lower().endswith(".csv"):
        frame = pd.read_csv(path_or_url)
    else:
        frame = pd.read_excel(path_or_url, header=1)
    frame.columns = [str(c).strip() for c in frame.columns]
    aliases = {
        "default.payment.next.month": "default payment next month",
        "default_payment_next_month": "default payment next month",
    }
    return frame.rename(columns=aliases)


def load_uci_default(path_or_url: str | Path = UCI_URL) -> DatasetBundle:
    """Map the public UCI benchmark into six monthly sequence tokens.

    SEX and AGE are excluded from model inputs and retained for audit. The source has
    no observation timestamp, so ``cohort`` is only a deterministic fold index. It
    must not be described as a temporal split.
    """
    frame = _read_frame(path_or_url)
    target_col = "default payment next month"
    required = [target_col, "LIMIT_BAL", "SEX", "AGE"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected UCI columns: {missing}")

    pay_cols = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]
    bill_cols = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
    paid_cols = ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]
    expected = pay_cols + bill_cols + paid_cols
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing monthly UCI columns: {missing}")

    limit = frame["LIMIT_BAL"].to_numpy(float).clip(min=1)
    bill = frame[bill_cols].to_numpy(float)
    paid = frame[paid_cols].to_numpy(float)
    status = frame[pay_cols].to_numpy(float)

    utilisation = np.clip(bill / limit[:, None], -1.0, 3.0)
    payment_ratio = np.clip(paid / np.maximum(np.abs(bill), 1.0), 0.0, 3.0)
    delinquency = np.clip(status, 0, 8) / 8.0
    balance_change = np.diff(utilisation, axis=1, prepend=utilisation[:, :1])
    zero_bill = (np.abs(bill) < 1.0).astype(float)
    sequence = np.stack(
        [utilisation, payment_ratio, delinquency, balance_change, zero_bill], axis=-1
    ).astype(np.float32)

    education = frame.get("EDUCATION", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    marriage = frame.get("MARRIAGE", pd.Series(np.zeros(len(frame)))).to_numpy(float)
    static = np.column_stack(
        [np.log1p(limit), education / 6.0, marriage / 3.0]
    ).astype(np.float32)

    age = frame["AGE"].to_numpy(float)
    age_band = pd.cut(
        age, bins=[17, 29, 44, 59, 100], labels=["18-29", "30-44", "45-59", "60+"]
    )
    sex_map = {1: "male", 2: "female"}
    audit = pd.DataFrame(
        {
            "sex": frame["SEX"].map(sex_map).fillna("unknown"),
            "age_band": age_band.astype(str),
            "age": age,
        }
    )
    fold_index = np.arange(len(frame)) % 10
    return DatasetBundle(
        sequence=sequence,
        static=static,
        target=frame[target_col].to_numpy(np.int64),
        exposure=(limit * np.clip(utilisation[:, -1], 0, 1.5)).astype(np.float32),
        cohort=fold_index.astype(np.int64),
        audit=audit[AUDIT_FEATURES + ["age"]],
        sequence_feature_names=[
            "utilisation", "payment_ratio", "delinquency", "balance_change", "zero_bill"
        ],
        static_feature_names=["log_credit_limit", "education_code", "marriage_code"],
    )
