import pytest

from decisionflow import DecisionFlow
from decisionflow.workflows.schema import INACTIVE

from workflow_fixtures import decision_model, workflow_document


def test_public_api_compiles_multistep_workflow_and_runs_exact_inference():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="enumeration",
    )

    compiled = flow.compile({"status": "open"})
    result = flow.infer({"status": "open"})

    assert len(compiled.worlds) == 6
    assert compiled.model_calls == 3
    assert result.valid_mass == pytest.approx(0.296)
    assert result.prediction == {
        "classify@0.route": "billing",
        "classify@0.fraud": False,
    }
    assert result.flow[0]["next_step"] == "resolved"
    assert result.diagnostics["selected_final_state"]["queue"] == "billing"
    assert result.marginals["review@1.action"][INACTIVE] > 0


def test_greedy_follows_local_decisions_and_reports_policy_conflict():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="greedy",
    )

    result = flow.infer({"status": "open"})

    assert result.prediction == {
        "classify@0.route": "billing",
        "classify@0.fraud": True,
        "review@1.action": "dismiss",
    }
    assert result.marginals == {}
    assert result.valid_mass is None
    assert result.diagnostics["prediction_feasible"] is False
    assert set(result.diagnostics["hard_constraint_violations"]) == {
        "fraud-routes-security",
        "fraud-cannot-be-dismissed",
    }


def test_backend_can_be_overridden_without_recompiling_the_workflow_definition():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="greedy",
    )

    result = flow.infer(
        {"status": "open"},
        backend="sampling",
        backend_options={"samples": 30_000, "seed": 4},
    )

    assert result.backend == "sampling"
    assert result.valid_mass == pytest.approx(0.296, abs=0.015)


def test_pc_lowering_matches_exact_workflow_enumeration():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="enumeration",
    )

    exact = flow.infer({"status": "open"})
    circuit = flow.infer({"status": "open"}, backend="pc")

    assert circuit.valid_mass == pytest.approx(exact.valid_mass)
    assert circuit.map_probability == pytest.approx(exact.map_probability)
    assert circuit.prediction == exact.prediction
    for variable, distribution in exact.marginals.items():
        assert circuit.marginals[variable] == pytest.approx(distribution)
    assert circuit.diagnostics["lowering"] == "categorical-world-sdd"


def test_dp_matches_enumeration_for_all_probabilistic_queries():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="enumeration",
    )

    exact = flow.infer({"status": "open"})
    dynamic = flow.infer({"status": "open"}, backend="dp")

    assert dynamic.valid_mass == pytest.approx(exact.valid_mass)
    assert dynamic.map_probability == pytest.approx(exact.map_probability)
    assert dynamic.prediction == exact.prediction
    for variable, distribution in exact.marginals.items():
        assert dynamic.marginals[variable] == pytest.approx(distribution)


def test_astar_and_wide_beam_recover_the_exact_map_flow():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="enumeration",
    )

    exact = flow.infer({"status": "open"})
    astar = flow.infer({"status": "open"}, backend="astar")
    beam = flow.infer(
        {"status": "open"},
        backend="beam",
        backend_options={"width": 16},
    )

    assert astar.prediction == exact.prediction
    assert astar.exact is True
    assert astar.valid_mass is None
    assert beam.prediction == exact.prediction
    assert beam.exact is False
    assert beam.diagnostics["width"] == 16


def test_sat_returns_a_feasible_flow_without_probability_queries():
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow_document(),
        backend="sat",
    )

    result = flow.infer({"status": "open"})

    assert result.prediction_kind == "satisfying_flow"
    assert result.diagnostics["prediction_feasible"] is True
    assert result.diagnostics["probabilities_used"] is False
    assert result.marginals == {}
    assert result.valid_mass is None


def test_soft_policies_match_across_enumeration_dp_pc_and_astar():
    workflow = workflow_document()
    workflow["constraints"]["soft"] = [
        {
            "name": "prefer-security-route",
            "penalty": 1.2,
            "expr": {"eq": [{"var": "classify.route"}, "security"]},
        }
    ]
    flow = DecisionFlow(
        model=decision_model,
        workflow=workflow,
        backend="enumeration",
    )

    exact = flow.infer({"status": "open"})
    dynamic = flow.infer({"status": "open"}, backend="dp")
    circuit = flow.infer({"status": "open"}, backend="pc")
    astar = flow.infer({"status": "open"}, backend="astar")

    assert dynamic.valid_mass == pytest.approx(exact.valid_mass)
    assert dynamic.map_probability == pytest.approx(exact.map_probability)
    assert circuit.valid_mass == pytest.approx(exact.valid_mass)
    assert circuit.map_probability == pytest.approx(exact.map_probability)
    assert dynamic.prediction == circuit.prediction == astar.prediction == exact.prediction


def test_workflow_validation_rejects_unknown_transition_target():
    workflow = workflow_document()
    workflow["steps"]["classify"]["transitions"][0]["goto"] = "missing"

    with pytest.raises(Exception, match="transition target does not exist"):
        DecisionFlow(model=decision_model, workflow=workflow)


def test_loops_are_unrolled_to_max_steps_and_keep_incomplete_mass_visible():
    workflow = {
        "name": "bounded-retry",
        "start": "retry",
        "max_steps": 2,
        "steps": {
            "retry": {
                "questions": [{"id": "done", "type": "noul"}],
                "transitions": [
                    {
                        "when": {"eq": [{"var": "done"}, True]},
                        "goto": "resolved",
                    },
                    {"otherwise": True, "goto": "retry"},
                ],
            },
            "resolved": {"terminal": True},
        },
    }

    flow = DecisionFlow(
        model=lambda request: {"done": {False: 0.5, True: 0.5}},
        workflow=workflow,
        backend="enumeration",
    )
    result = flow.infer({})
    dynamic = flow.infer({}, backend="dp")

    assert result.valid_mass == pytest.approx(0.75)
    assert dynamic.valid_mass == pytest.approx(result.valid_mass)
    for variable, distribution in result.marginals.items():
        assert dynamic.marginals[variable] == pytest.approx(distribution)
    assert result.diagnostics["world_count"] == 3
    assert result.diagnostics["complete_world_count"] == 2
