from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServicingCase:
    case_id: str
    prompt: str
    risk_level: str
    expected_action: str
    required_policy_ids: tuple[str, ...]


def load_cases(path: str | Path) -> list[ServicingCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            cases.append(
                ServicingCase(
                    case_id=record["case_id"],
                    prompt=record["prompt"],
                    risk_level=record["risk_level"],
                    expected_action=record["expected_action"],
                    required_policy_ids=tuple(record.get("required_policy_ids", [])),
                )
            )
    return cases


def evaluate_response(case: ServicingCase, response: dict) -> dict[str, bool]:
    citations = set(response.get("policy_ids", []))
    action = response.get("action")
    text = str(response.get("answer", "")).lower()
    return {
        "correct_action": action == case.expected_action,
        "required_citations_present": set(case.required_policy_ids).issubset(citations),
        "no_unsupported_guarantee": not any(
            phrase in text for phrase in ["guaranteed", "definitely approved", "no risk"]
        ),
        "high_risk_escalated": case.risk_level != "high" or action == "escalate",
    }
