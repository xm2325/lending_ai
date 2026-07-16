# Model card

## Model purpose

Predict the probability of a future default proxy from monthly account behaviour and a small set of non-protected static fields.

## Models

- Logistic regression using the latest month.
- Histogram gradient boosting using flattened history.
- LSTM fusion model.
- Transformer encoder fusion model.

## Data

The repository default is synthetic and contains no customer records. An optional adapter maps the public UCI credit-card default dataset into six monthly tokens. The adapter keeps sex and age out of training and uses them only for audit.

## Main metrics

ROC AUC, Gini, average precision, Brier score, expected calibration error, recall at top 10%, and exposure capture at top 10%.

## Known limits

- Synthetic outputs are engineering and analysis tests, not evidence of real lending performance.
- The UCI target and customer population do not match a UK lender's production setting.
- Historical labels can reflect past policy and selection effects.
- Reject inference, macroeconomic stress, treatment effects and customer outcomes need separate work.
- A risk score is not a credit decision or customer communication.

## Minimum release checks

Time-based validation, champion comparison, calibration, subgroup audit, drift, stress tests, latency, reproducible build, rollback, and independent review.
