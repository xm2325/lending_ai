# Real-data card: UCI Default of Credit Card Clients

The repository supports the public UCI Default of Credit Card Clients benchmark: 30,000 credit-card clients, six monthly bill amounts, repayment amounts and repayment-status fields, plus a next-month default label.

## Valid use in this project

The data can test whether six-month account history improves customer-level default ranking, how performance changes with history length, whether probabilities are calibrated under cross-validation, and whether errors differ by sex/age audit groups.

## Invalid claims

The dataset is not UK Capital One data. It has no observation date, application decision, rejection population, interest rate, income, bureau history, intervention, loss given default or customer-service text. Row order is not time. It cannot support a temporal drift claim, UK underwriting policy, causal treatment effect, fair-lending conclusion, profit estimate or production release.

## Validation design

The `uci` command uses customer-level cross-validation and customer-level bootstrap confidence intervals. It explicitly labels this as non-temporal validation. A production study must replace it with chronological train/calibration/test cohorts and point-in-time feature reconstruction.

## Data provenance

Primary source: UCI Machine Learning Repository, dataset 350, donated by I-Cheng Yeh. The checked-in CSV is only a 20-row parser fixture copied from the public dataset; it must not be used for performance claims. Download the complete source before running the benchmark.
