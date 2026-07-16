from lending_ai_lab.llm_eval import ServicingCase, evaluate_response


def test_high_risk_case_requires_escalation_and_citation():
    case = ServicingCase("x", "help", "high", "escalate", ("POLICY_1",))
    result = evaluate_response(
        case,
        {"action": "escalate", "policy_ids": ["POLICY_1"], "answer": "A colleague will help."},
    )
    assert all(result.values())
