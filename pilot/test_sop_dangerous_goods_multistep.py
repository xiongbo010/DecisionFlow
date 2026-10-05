"""Tests for multi-step dangerous-goods policy grounding."""

from __future__ import annotations

import csv

from sop_dangerous_goods_multistep import (
    ACTIONS,
    DATA_PATH,
    StepCircuit,
    allowed_actions,
    distributions,
    final_class,
    possible_classes,
    procedural_from_masked,
    ready_for_final,
    value_from_row,
)
from sop_dangerous_goods_probjev import CLASS_VALUES, normalize


def row(product_id: str):
    with DATA_PATH.open(newline="") as handle:
        return next(r for r in csv.DictReader(handle) if r["product_id"] == product_id)


def test_grounding_progresses_to_unique_final_class() -> None:
    item = row("P_13181")
    observations = {}
    assert not ready_for_final(item, observations)
    assert allowed_actions(item, observations) == set(ACTIONS[:-1])
    for action in ACTIONS[:-1]:
        observations[action] = value_from_row(item, action)
    assert ready_for_final(item, observations)
    assert allowed_actions(item, observations) == {"finalize"}
    assert possible_classes(item, observations) == {"Hazard Class B"}
    assert final_class(item, observations) == "Hazard Class B"


def test_invalid_id_finalizes_without_tools() -> None:
    item = row("P1_3191")
    assert ready_for_final(item, {})
    assert allowed_actions(item, {}) == {"finalize"}
    assert possible_classes(item, {}) == {"Unable to Decide"}


def test_step_circuit_returns_legal_joint_map() -> None:
    local = {
        "action": normalize({action: index + 1 for index, action in enumerate(ACTIONS)}),
        "ready": normalize({False: 0.8, True: 0.2}),
        "class": normalize({label: index + 1 for index, label in enumerate(CLASS_VALUES)}),
    }
    circuit = StepCircuit.build(
        {ACTIONS[0], ACTIONS[1]}, False, {"Hazard Class A", "Hazard Class B"}
    )
    result = circuit.infer(local)
    assert result["joint_map"]["action"] in {ACTIONS[0], ACTIONS[1]}
    assert result["joint_map"]["ready"] is False
    assert result["joint_map"]["class"] in {"Hazard Class A", "Hazard Class B"}


def test_procedural_baseline_reconstructs_tool_results() -> None:
    item = row("P_13181")
    trace = []
    observations = {}
    for action in ACTIONS[:-1]:
        trace.append(
            {
                "observations_before": dict(observations),
                "selected": {"action": action, "class": "Unable to Decide"},
                "selected_joint_is_consistent": False,
            }
        )
        observations[action] = value_from_row(item, action)
    trace.append(
        {
            "observations_before": dict(observations),
            "selected": {"action": "finalize", "class": "Unable to Decide"},
            "selected_joint_is_consistent": False,
        }
    )
    masked = {
        "success": False,
        "failure": "wrong_class",
        "trace": trace,
        "final_class": "Unable to Decide",
    }
    result = procedural_from_masked(item, masked)
    assert result["success"]
    assert result["final_class"] == "Hazard Class B"
    assert result["trace"][-1]["selected_joint_is_consistent"]
