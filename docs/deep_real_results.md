# Deep sequence benchmark on real public credit data

This report is generated from shared customer-level folds on the official UCI dataset. It is a research benchmark, not a production underwriting validation.

## Decision first

- Tabular champion: **Histogram gradient boosting** (ROC AUC 0.7839)
- Best deep challenger: **TCN** (ROC AUC 0.7747)
- Technical gate: **retain_tabular_champion**
- Overall best model: **Histogram gradient boosting**
- Calibration uses fold-local validation customers; test customers are not used to fit the calibrator.
- Production use remains blocked because the public source has no point-in-time UK policy data.

## Shared-fold comparison

| model                       |   roc_auc |   average_precision |   brier_score |   ece_10 |   calibration_intercept |   calibration_slope |   recall_at_10pct |   exposure_capture_at_10pct |   training_seconds |
|:----------------------------|----------:|--------------------:|--------------:|---------:|------------------------:|--------------------:|------------------:|----------------------------:|-------------------:|
| Histogram gradient boosting |    0.7839 |              0.5557 |        0.1338 |   0.0052 |                  0.0093 |              0.9993 |            0.3159 |                      0.4084 |             1.0342 |
| TCN                         |    0.7747 |              0.5441 |        0.1365 |   0.0073 |                  0.0243 |              1.0102 |            0.3064 |                      0.3491 |             8.7895 |
| LSTM                        |    0.7711 |              0.5268 |        0.1378 |   0.0081 |                  0.0181 |              1.007  |            0.2963 |                      0.3419 |             6.474  |
| Transformer                 |    0.7709 |              0.5179 |        0.1391 |   0.0053 |                  0.0183 |              1.0113 |            0.291  |                      0.3223 |             9.7885 |

## Paired customer-level uncertainty

Positive values mean the deep challenger improves on the tabular champion. For Brier score, the sign is reversed so positive still means better calibration error.

| metric                    |   improvement |   ci_low |   ci_high |   probability_improvement |
|:--------------------------|--------------:|---------:|----------:|--------------------------:|
| roc_auc                   |       -0.0093 |  -0.0125 |   -0.006  |                         0 |
| average_precision         |       -0.0117 |  -0.0176 |   -0.005  |                         0 |
| brier_score               |       -0.0027 |  -0.0034 |   -0.002  |                         0 |
| exposure_capture_at_10pct |       -0.0593 |  -0.0771 |   -0.0424 |                         0 |

## Does the neural model use month order?

| control                       |   roc_auc |   average_precision |   brier_score |   ece_10 |   calibration_intercept |   calibration_slope |   recall_at_10pct |   exposure_capture_at_10pct |
|:------------------------------|----------:|--------------------:|--------------:|---------:|------------------------:|--------------------:|------------------:|----------------------------:|
| baseline                      |    0.7747 |              0.5441 |        0.1365 |   0.0073 |                  0.0243 |              1.0102 |            0.3064 |                      0.3491 |
| reversed_history              |    0.7409 |              0.4695 |        0.1497 |   0.0359 |                 -0.0521 |              0.8485 |            0.2548 |                      0.2551 |
| latest_month_unavailable      |    0.7427 |              0.4842 |        0.1458 |   0.0243 |                 -0.0008 |              0.8855 |            0.2829 |                      0.3005 |
| latest_payment_feed_missing   |    0.774  |              0.5434 |        0.1366 |   0.0087 |                  0.0134 |              1.0015 |            0.3064 |                      0.3488 |
| delinquency_feed_missing      |    0.6139 |              0.2895 |        0.1776 |   0.097  |                  0.2904 |              0.777  |            0.1451 |                      0.0667 |
| latest_utilisation_plus_20pct |    0.7735 |              0.5434 |        0.1367 |   0.0098 |                  0.0421 |              1.034  |            0.3056 |                      0.339  |
| shared_random_permutation     |    0.7534 |              0.4882 |        0.1456 |   0.0252 |                 -0.1027 |              0.8683 |            0.275  |                      0.2995 |

## Input and feed sensitivity

These are controlled perturbations, not macroeconomic forecasts. They test how model scores and the top-10% review population respond when important feeds are missing or shifted.

| scenario                      |   mean_score |   mean_score_shift |   top_10pct_overlap |   roc_auc |   brier_score |
|:------------------------------|-------------:|-------------------:|--------------------:|----------:|--------------:|
| baseline                      |       0.2194 |             0      |              1      |    0.7747 |        0.1365 |
| reversed_history              |       0.2024 |            -0.017  |              0.623  |    0.7409 |        0.1497 |
| latest_month_unavailable      |       0.2021 |            -0.0172 |              0.8203 |    0.7427 |        0.1458 |
| latest_payment_feed_missing   |       0.2196 |             0.0002 |              0.997  |    0.774  |        0.1366 |
| delinquency_feed_missing      |       0.1242 |            -0.0951 |              0.1033 |    0.6139 |        0.1776 |
| latest_utilisation_plus_20pct |       0.2204 |             0.0011 |              0.979  |    0.7735 |        0.1367 |
| shared_random_permutation     |       0.2148 |            -0.0046 |              0.7583 |    0.7534 |        0.1456 |

## Automated data checks

| check                 | severity   | status   | observed   | expectation                          |
|:----------------------|:-----------|:---------|:-----------|:-------------------------------------|
| required_columns      | fail       | pass     | none       | all required fields present          |
| row_count             | warn       | pass     | 30000      | 30,000 rows for unchanged UCI source |
| duplicate_customer_id | fail       | pass     | 0          | 0 duplicate IDs                      |
| missing_cells         | fail       | pass     | 0          | 0 missing cells                      |
| binary_target         | fail       | pass     | 0          | target values only 0 or 1            |
| target_prevalence     | warn       | pass     | 0.2212     | between 0.05 and 0.50                |
| positive_credit_limit | fail       | pass     | 0          | all credit limits positive           |
| plausible_age         | fail       | pass     | 0          | 18 <= age <= 100                     |
| sex_code_domain       | warn       | pass     | 0          | codes in {1,2}                       |
| nonnegative_payments  | warn       | pass     | 0          | payment amounts nonnegative          |

## Release blockers

- no point-in-time application timestamps
- no UK population or policy data
- no rejected-applicant outcomes
- no loss-given-default or treatment-effect evidence
- no independent model validation
