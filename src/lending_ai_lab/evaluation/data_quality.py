from __future__ import annotations

from pathlib import Path

import pandas as pd

from lending_ai_lab.data.uci_default import _read_frame


REQUIRED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default payment next month",
]


def inspect_uci_source(path: str | Path) -> pd.DataFrame:
    frame = _read_frame(path)
    checks: list[dict[str, object]] = []

    def add(name: str, severity: str, passed: bool, observed: object, expectation: str) -> None:
        checks.append(
            {
                "check": name,
                "severity": severity,
                "status": "pass" if passed else severity,
                "observed": observed,
                "expectation": expectation,
            }
        )

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    add("required_columns", "fail", not missing, ",".join(missing) or "none", "all required fields present")
    add("row_count", "warn", len(frame) == 30_000, len(frame), "30,000 rows for unchanged UCI source")

    if "ID" in frame:
        duplicate_ids = int(frame["ID"].duplicated().sum())
        add("duplicate_customer_id", "fail", duplicate_ids == 0, duplicate_ids, "0 duplicate IDs")
    missing_cells = int(frame.isna().sum().sum())
    add("missing_cells", "fail", missing_cells == 0, missing_cells, "0 missing cells")

    target = "default payment next month"
    if target in frame:
        invalid = int((~frame[target].isin([0, 1])).sum())
        add("binary_target", "fail", invalid == 0, invalid, "target values only 0 or 1")
        rate = float(frame[target].mean())
        add("target_prevalence", "warn", 0.05 <= rate <= 0.50, round(rate, 6), "between 0.05 and 0.50")

    if "LIMIT_BAL" in frame:
        invalid = int((frame["LIMIT_BAL"] <= 0).sum())
        add("positive_credit_limit", "fail", invalid == 0, invalid, "all credit limits positive")
    if "AGE" in frame:
        invalid = int(((frame["AGE"] < 18) | (frame["AGE"] > 100)).sum())
        add("plausible_age", "fail", invalid == 0, invalid, "18 <= age <= 100")
    if "SEX" in frame:
        invalid = int((~frame["SEX"].isin([1, 2])).sum())
        add("sex_code_domain", "warn", invalid == 0, invalid, "codes in {1,2}")

    payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]
    if set(payment_columns).issubset(frame.columns):
        negative = int((frame[payment_columns] < 0).sum().sum())
        add("nonnegative_payments", "warn", negative == 0, negative, "payment amounts nonnegative")

    return pd.DataFrame(checks)
