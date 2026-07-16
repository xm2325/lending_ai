# Lending AI Lab

A reproducible, business-first workbench for credit-risk modelling on monthly account histories and safe evaluation of LLM servicing products.

> **Evidence boundary:** the checked-in real dataset is the public UCI Default of Credit Card Clients dataset. It does not represent Capital One, the United Kingdom, current lending policy or production performance.

## Decision first

The strongest current real-data result is **not** a deep-learning win.

| Model | ROC AUC | Average precision | Brier score | Recall at top 10% | Exposure capture at top 10% |
|---|---:|---:|---:|---:|---:|
| Histogram gradient boosting | 0.7839 | 0.5557 | 0.1338 | 0.3159 | 0.4084 |
| TCN | 0.7747 | 0.5441 | 0.1365 | 0.3064 | 0.3491 |
| LSTM | 0.7711 | 0.5268 | 0.1378 | 0.2963 | 0.3419 |
| Transformer | 0.7709 | 0.5179 | 0.1391 | 0.2910 | 0.3223 |

The best deep challenger is the temporal convolutional network (TCN), but paired customer-level bootstrap results favour the tabular champion:

```text
ROC AUC improvement:                   -0.0093  (95% CI -0.0125 to -0.0060)
Average-precision improvement:         -0.0117  (95% CI -0.0176 to -0.0050)
Brier-score improvement:               -0.0027  (95% CI -0.0034 to -0.0020)
Top-10% exposure-capture improvement:  -0.0593  (95% CI -0.0771 to -0.0424)
```

The technical decision is therefore:

> **Retain histogram gradient boosting as champion. Do not promote the neural challenger.**

This is a more useful deep-learning result than selecting a neural model because the job title mentions deep learning. The project shows how to test whether sequence models add decision value and how to stop them when they do not.

## What the project answers

The workbench addresses the questions a lending data scientist is likely to face:

- Does six-month history add value over the latest month alone?
- Is a neural sequence model better than a strong tree model using the same information?
- Are probability estimates calibrated on customers not used to fit the calibrator?
- Does the neural model use month order, or only summary statistics?
- What happens when the latest month, payment feed or delinquency feed is unavailable?
- How many risky customers and how much default-linked exposure are captured at a fixed review capacity?
- Are results stable across audit groups?
- Which data failures should warn, degrade to a fallback, or stop scoring?
- Is the evidence sufficient for shadow testing, or should the current champion remain in place?
- Which conclusions cannot be made from public data?

## Real-data findings

### Account history adds signal

With logistic regression, ROC AUC rises from 0.7399 with one month to 0.7571 with six months. The improvement is gradual rather than coming from one arbitrary history length.

### Sequence order matters to the TCN

The TCN ROC AUC falls from 0.7747 to 0.7409 when history is reversed, and to 0.7534 under a shared random month permutation. This indicates that the model uses temporal order, although that use does not produce better performance than the tree champion.

### Delinquency is a critical feed

Removing delinquency inputs reduces TCN ROC AUC from 0.7747 to 0.6139 and changes 89.7% of the customers in the top-10% review population. A production version should fail closed, use an approved fallback, or stop customer action when this feed is unavailable.

### Latest-month availability matters

Removing the latest month reduces TCN ROC AUC to 0.7427. This supports explicit history-completeness checks and separate cold-start policies.

### Calibration is fold-local

Each outer test fold is scored by a model trained on other customers. Platt calibration is fitted only on an inner validation subset, not on the outer test customers. The generated report includes expected calibration error, calibration intercept and calibration slope.

## Repository structure

```text
src/lending_ai_lab/
  data/                  UCI and synthetic adapters
  models/                logistic, tree, LSTM, GRU, TCN and Transformer models
  evaluation/            calibration, paired uncertainty, drift, fairness and data checks
  deep_real_experiment.py
  real_experiment.py
  experiment.py
  serving/               FastAPI contract and fail-closed loading

artifacts_real/           tabular real-data benchmark
artifacts_deep/           shared-fold deep benchmark and promotion gate
docs/                     generated decision reports and design records
site/                     GitHub Pages review dashboards
tests/                    unit and integration checks
.github/workflows/        CI, real-data benchmark, deep benchmark and Pages deployment
```

## Reproduce the real benchmark

The official UCI XLS file, licence record and SHA-256 checksum are stored under `data/raw/` and were retrieved by GitHub Actions under CC BY 4.0.

```bash
python -m pip install -e ".[dev,uci]"

python -m lending_ai_lab.cli uci \
  --data-path data/raw/default_of_credit_card_clients.xls \
  --output-dir artifacts_real

python -m lending_ai_lab.cli deep-uci \
  --data-path data/raw/default_of_credit_card_clients.xls \
  --output-dir artifacts_deep \
  --folds 3 \
  --epochs 6
```

Generate the reports and dashboard:

```bash
python scripts/build_real_report.py \
  --artifacts artifacts_real \
  --output docs/real_results.md

python scripts/build_deep_report.py \
  --artifacts artifacts_deep \
  --output docs/deep_real_results.md

python scripts/build_real_dashboard.py \
  --artifacts artifacts_deep \
  --output site/real/index.html
```

## Generated evidence

```text
artifacts_real/real_model_metrics.csv
artifacts_real/real_history_ablation.csv
artifacts_real/real_sequence_controls.csv
artifacts_real/real_subgroup_metrics.csv
artifacts_real/real_summary.json

artifacts_deep/deep_model_metrics.csv
artifacts_deep/deep_fold_metrics.csv
artifacts_deep/deep_paired_differences.csv
artifacts_deep/deep_calibration_bins.csv
artifacts_deep/deep_order_sensitivity.csv
artifacts_deep/deep_stress_tests.csv
artifacts_deep/data_quality_checks.csv
artifacts_deep/promotion_gate.json
artifacts_deep/deep_summary.json

docs/real_results.md
docs/deep_real_results.md
site/real/index.html
```

## Automated controls

GitHub Actions performs the following chain:

```text
verify UCI SHA-256 and attribution
        -> lint and tests
        -> shared customer folds
        -> fold-local calibration
        -> paired bootstrap uncertainty
        -> order and feed sensitivity
        -> promotion gate
        -> generated report and dashboard
        -> committed reproducible evidence
```

The real source passed all hard automated checks: required fields, row count, duplicate IDs, missing cells, binary target, positive credit limits, plausible age, valid sex codes and non-negative payment amounts.

## LLM servicing safety module

`data/sample/llm_servicing_cases.jsonl` and `lending_ai_lab.llm_eval` provide an offline safety harness for a servicing assistant. The tests require escalation for high-risk requests, approved policy citations for factual claims and failure on unsupported guarantees. Deterministic systems remain the source of truth for balances, fees, eligibility, account actions and credit decisions.

## Production blockers

This public benchmark cannot support production promotion because it lacks:

- point-in-time application timestamps;
- UK customers and policy context;
- rejected-applicant outcomes;
- loss-given-default and treatment-effect evidence;
- macroeconomic and policy-change cohorts;
- independent model validation.

A production programme would next require point-in-time internal data, temporal outcome validation, reject-inference analysis, customer-impact review, approved reason codes, monitoring thresholds, a fallback model and a tested rollback path.
