# LLM servicing product review

The role also supports teams building LLM-based servicing products. This module treats that work as a product and risk review problem rather than a chatbot demo.

## Proposed architecture

1. Classify the customer request and detect high-risk topics.
2. Redact or mask unnecessary personal data before retrieval.
3. Retrieve only approved, versioned policy passages.
4. Generate an answer constrained by the retrieved passages.
5. Return policy identifiers with every factual answer.
6. Escalate hardship, vulnerability, fraud, complaint, legal, approval and credit-reporting cases.
7. Store prompt, retrieved passages, output, action and model version in an audit record.

## Offline evaluation

The JSONL test set checks action routing, required citations, unsupported guarantees and high-risk escalation. A production evaluation would also measure groundedness, policy-version freshness, PII leakage, refusal quality, latency and human-review agreement.

## Launch gate

Do not launch based on general language quality. Launch only after high-risk routing has very high recall, answers are grounded in approved text, policy updates invalidate stale responses, and the product has a human fallback.
