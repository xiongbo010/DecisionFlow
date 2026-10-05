import json
from io import StringIO

from decisionflow_experiments import (
    EXPERIMENTS,
    ExperimentRunner,
    TrajectoryExperimentRunner,
)


def probabilities(spec):
    rows = {}
    for question in spec.questions:
        if question["type"] == "noul":
            rows[question["id"]] = 0.4
        else:
            rows[question["id"]] = [1.0] * len(question["options"])
    return rows


def test_all_fixed_experiments_use_public_api():
    for spec in EXPERIMENTS.values():
        if spec.record_mode != "fixed":
            continue
        result = ExperimentRunner.create(spec, backend="auto").run_record(
            {"id": "test", "state": {}, "probabilities": probabilities(spec)}
        )
        assert result["experiment"] == spec.name
        assert result["inference"]["exact"] is True


def test_grounded_experiment_accepts_per_record_program():
    spec = EXPERIMENTS["typed_decisions"]
    record = {
        "id": "grounded",
        "request": {
            "state": {},
            "questions": [
                {"id": "action", "type": "choice", "options": ["allow", "deny"]}
            ],
        },
        "probabilities": {"action": [0.9, 0.1]},
        "constraints": {
            "hard": [{"expr": {"eq": [{"var": "action"}, "deny"]}}]
        },
    }
    result = ExperimentRunner.create(spec, backend="enumeration").run_record(record)
    assert result["joint_map"] == {"action": "deny"}


def test_multistep_runner_uses_trajectory_api():
    record = {
        "id": "ticket",
        "trajectory": {
            "initial_state": "new",
            "actions": ["inspect", "resolve"],
            "horizon": 2,
            "terminal_states": ["done"],
            "policy": {
                "new": {"inspect": 0.7, "resolve": 0.3},
                "inspected": {"inspect": 0.1, "resolve": 0.9},
            },
            "transitions": {
                "new": {"inspect": "inspected"},
                "inspected": {"resolve": "done"},
            },
        },
    }
    result = TrajectoryExperimentRunner.create().run_record(record)
    assert result["trajectory_map"][0] == ["inspect", "inspected"]
    assert result["first_action_marginals"]["inspect"] == 1.0
