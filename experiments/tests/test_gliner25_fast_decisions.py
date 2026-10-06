from decisionflow_experiments.gliner25_fast_decisions import (
    accepted_rules,
    audit_rules,
    _df_constraints,
    _df_request,
    _gold,
    _rule_holds,
)


def _row(heads):
    return {"input": "example", "output": {"classifications": heads}}


def test_cardinality_rule_is_audited_and_translated():
    row = _row([
        {
            "task": "feedback_type",
            "labels": ["bug", "praise"],
            "true_label": ["bug"],
            "multi_label": False,
        },
        {
            "task": "product_area",
            "labels": ["mobile", "billing"],
            "true_label": ["mobile"],
            "multi_label": True,
        },
    ])
    audit = audit_rules([row], "product_feedback")
    assert audit[0]["gold_violations"] == 0
    rules = accepted_rules([row], "product_feedback")
    request = _df_request(row, "product_feedback")
    assert {q["id"] for q in request["questions"]} == {
        "feedback_type", "product_area::mobile", "product_area::billing"
    }
    constraints = _df_constraints(row, "product_feedback", rules)
    assert constraints["hard"][0]["expr"]["at_least"]["k"] == 1


def test_semantic_rule_rejected_when_gold_has_counterexample():
    rows = [
        _row([
            {"task": "result", "labels": ["upcoming", "win"], "true_label": ["upcoming"], "multi_label": False},
            {"task": "upset", "labels": ["yes", "no"], "true_label": ["yes"], "multi_label": False},
            {"task": "sport", "labels": ["soccer", "other"], "true_label": ["soccer"], "multi_label": False},
        ])
    ]
    audit = audit_rules(rows, "sports_recap")
    assert audit[0]["gold_violations"] == 1
    assert "upcoming-is-not-upset" not in {r.name for r in accepted_rules(rows, "sports_recap")}


def test_handoff_auxiliary_gold_satisfies_iff():
    row = _row([
        {
            "task": "should_handoff",
            "labels": ["yes", "no"],
            "true_label": ["yes"],
            "multi_label": False,
        }
    ])
    gold = _gold(row, structured_handoff=True)
    rule = accepted_rules([row], "agent_handoff")[0]
    assert gold["policy_scope"] == {"handoff_required"}
    assert _rule_holds(rule, gold)
