# Decision brief

## Business question

Can monthly account behaviour improve default-risk ranking beyond a strong non-sequential model, early enough to support a controlled customer treatment?

## Decision rule

A sequence model is not promoted because it has a higher test AUC once. It must pass six gates:

1. Out-of-time ROC AUC and Gini improve over the strongest baseline.
2. Recall and exposure capture improve within a fixed review capacity.
3. The model gives useful risk signals with fewer than six observed months.
4. Calibration and expected-value proxies are acceptable at the proposed threshold.
5. Subgroup and drift checks do not show an unexplained failure.
6. Inference, tests, versioning and rollback requirements are met.

## Intended decision

The model produces a risk score for controlled research and shadow testing. It does not approve or decline credit by itself. Any customer action requires a separately reviewed policy layer.

## Stakeholder outputs

- Risk: discrimination, calibration, stress and subgroup evidence.
- Product: capacity-constrained recall and customer-treatment trade-offs.
- Engineering: data contract, latency, model artifact and rollback plan.
- Compliance: feature policy, audit log and explanation route.
- Senior leadership: one-page recommendation with assumptions and limits.
