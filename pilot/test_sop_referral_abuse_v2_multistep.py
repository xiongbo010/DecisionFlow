"""Tests for the SOP-Bench referral-abuse v2 ProbJev pilot."""

from __future__ import annotations

import csv

from sop_dangerous_goods_probjev import normalize
from sop_referral_abuse_v2_multistep import (
    ACTIONS,
    DATA_PATH,
    ENFORCEMENTS,
    RISKS,
    VIOLATIONS,
    StepCircuit,
    audit,
    derive,
    semantic_triples,
)


def rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_sop_rules_reproduce_all_gold_decisions() -> None:
    result = audit(rows())
    assert result["rows"] == 200
    assert result["matches"] == 200
    assert result["mismatches"] == []


def test_representative_critical_case() -> None:
    item = next(row for row in rows() if row["final_decision"] == "Permanent Account Closure")
    violation, risk, action = derive(item)
    assert violation not in {"No Violation", "Inconclusive", "Personal Orders"}
    assert risk == "CRITICAL"
    assert action == "Permanent Account Closure"


def test_circuit_joint_map_satisfies_cross_type_relation() -> None:
    local = {
        "action": normalize({value: index + 1 for index, value in enumerate(ACTIONS)}),
        "ready": normalize({False: 0.8, True: 0.2}),
        "violation": normalize({value: index + 1 for index, value in enumerate(VIOLATIONS)}),
        "risk": normalize({value: index + 1 for index, value in enumerate(RISKS)}),
        "enforcement": normalize({value: index + 1 for index, value in enumerate(ENFORCEMENTS)}),
    }
    actions = set(ACTIONS[:-1])
    triples = semantic_triples()
    result = StepCircuit.build(actions, False, triples).infer(local)
    joint = result["joint_map"]
    assert joint["action"] in actions
    assert joint["ready"] is False
    assert (joint["violation"], joint["risk"], joint["enforcement"]) in triples
    assert 0 < result["z"] <= 1
