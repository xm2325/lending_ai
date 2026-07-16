# Lending AI Lab

A business-first, reproducible workbench for sequence-aware credit-risk modelling and safe LLM servicing product review.

> **Status:** reproducible public-data workbench. GitHub Actions downloads the official UCI dataset, records its licence and checksum, runs the real benchmark, and commits generated results. It does not represent Capital One data, policy or performance.

## Executive question

**Does monthly account history add enough early-warning value over a strong tree baseline to justify a controlled shadow test?**

The repository compares a latest-month logistic model, a flattened-history tree model, an LSTM fusion model and a Transformer fusion model. The recommendation is based on more than ROC AUC: it checks fixed-capacity recall, exposure capture, calibration, early detection, subgroup results, drift and latency.

## Why this project fits the role

| Role requirement | Project evidence |
|---|---|
| Deep learning for underwriting | LSTM and Transformer models over monthly credit behaviour |
| Sequential data | Six-month utilisation, payment, delinquency, balance-change and cash-share tokens |
| Structured and unstructured product work | Credit sequence pipeline plus an LLM servicing evaluation pack |
| Pre-training / fine-tuning understanding | The LLM design separates foundation-model choice from domain adaptation and offline product tests |
| PyTorch and maintainable Python | Typed package, tests, CLI, CI, FastAPI contract and reproducible artifacts |
| Business communication | Decision brief, model card, release gate and interactive review dashboard |
| Cross-functional work | Separate outputs for risk, product, engineering, compliance and leadership |

## Project flow

```text
monthly account data + approved static fields
                    |
              time-based split
                    |
       baseline     |     sequence models
    logistic/tree   |    LSTM/Transformer
                    |
   discrimination + calibration + early detection
                    |
     subgroup + drift + latency + value proxy
                    |
       keep champion / shadow test / stop
```

## Run

```bash
python -m pip install -e ".[dev]"
python -m lending_ai_lab.cli demo --n-samples 12000 --epochs 8 --output-dir artifacts
pytest -q
```

Open `site/index.html` after the experiment.

## Outputs

```text
artifacts/model_metrics.csv
artifacts/prefix_performance.csv
artifacts/month_occlusion.csv
artifacts/subgroup_metrics.csv
artifacts/feature_drift.csv
artifacts/threshold_value_curve.csv
artifacts/latency.csv
artifacts/summary.json
site/index.html
```

## Real-data adapter

`load_uci_default()` maps the public UCI default-of-credit-card-clients data into six monthly tokens. Install the optional Excel dependency and pass a local file path:

```python
from lending_ai_lab.data.uci_default import load_uci_default

data = load_uci_default("data/raw/default_of_credit_card_clients.xls")
```

The UCI adapter is for a public benchmark, not a claim that its population, labels or policy match UK underwriting.

## Current synthetic smoke-test finding

The first reproducible run does **not** force a deep-learning win. The latest-month logistic model remains the non-sequential champion on overall discrimination, while the sequence challenger can capture more default-linked exposure inside a fixed top-10% review capacity. The proposed action is therefore champion/challenger shadow testing, not automatic replacement. This is a stronger lending-science story than selecting a neural model only because the role mentions deep learning.

## Key design choices

Protected audit fields are excluded from model inputs. Data are split by time cohort rather than randomly. Deep models receive random history truncation during training so that early-warning curves can be tested. The tree model receives the full flattened sequence, making it a harder baseline than a latest-month-only comparison.

The business-value curve is labelled as a proxy because real loss given default, treatment effect, operational cost and customer outcome data are unavailable. This avoids turning an assumed cost matrix into a false profit claim.

## LLM servicing module

`data/sample/llm_servicing_cases.jsonl` and `lending_ai_lab.llm_eval` define a small offline test harness for an LLM servicing assistant. High-risk requests must be escalated, factual answers must cite approved policy identifiers, and unsupported guarantees fail evaluation. See `docs/llm_servicing_design.md`.

## Next production-grade work

Replace synthetic data with approved, de-identified internal data; define a point-in-time feature store; test reject-inference and policy-selection bias; add macroeconomic stress cohorts; calibrate using later cohorts; estimate treatment effects before customer action; and document independent model validation.

## v0.2: real-data benchmark

The repository now separates two evidence levels:

1. `demo`: a fully runnable synthetic time-cohort experiment for testing the production workflow.
2. `uci`: a real public-data benchmark using all 30,000 UCI customers after the user downloads the official file or an unchanged CSV mirror.

Run the real benchmark:

```bash
python -m pip install -e ".[dev,uci]"
python -m lending_ai_lab.cli uci \
  --data-path data/raw/UCI_Credit_Card.csv \
  --output-dir artifacts_real
```

Outputs:

```text
artifacts_real/real_model_metrics.csv
artifacts_real/real_history_ablation.csv
artifacts_real/real_subgroup_metrics.csv
artifacts_real/real_sequence_controls.csv
artifacts_real/real_summary.json
```

The real benchmark compares latest-month logistic regression, full-history logistic regression and histogram gradient boosting with customer-level cross-validation, bootstrap confidence intervals, history-length ablation, sequence-order controls and subgroup audit. It does **not** call the split temporal because UCI does not provide observation dates.

The checked-in `data/sample/UCI_Credit_Card_sample.csv` contains only 20 authentic public rows and is used solely as a parser fixture. It is too small for performance reporting.

## v0.3: automated real-data evidence

The `real-data-benchmark` GitHub Action performs the complete reproducibility chain:

```text
official UCI ZIP -> unchanged XLS -> SHA-256 + source record
                 -> tests -> real-data benchmark -> generated report
                 -> Actions artifact + committed results
```

The checked-in source is permitted by the dataset's CC BY 4.0 licence and is accompanied by `data/raw/SOURCE.md`. The generated decision report is `docs/real_results.md`.
