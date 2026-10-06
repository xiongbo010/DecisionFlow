from decisionflow import DecisionFlowTools

from workflow_fixtures import decision_model, workflow_document


def test_tool_runs_configured_workflow():
    tools = DecisionFlowTools(
        model=decision_model,
        workflow=workflow_document(),
        backend="enumeration",
    )

    result = tools.call("decisionflow_run", {"state": {"status": "open"}})

    assert result["prediction_kind"] == "joint_map"
    assert result["valid_mass"] > 0
    assert len(tools.schemas) == 2


def test_tool_accepts_workflow_and_backend_per_call():
    tools = DecisionFlowTools(model=decision_model)

    result = tools.call(
        "decisionflow_run",
        {
            "state": {"status": "open"},
            "workflow": workflow_document(),
            "backend": "greedy",
        },
    )

    assert result["prediction_kind"] == "local_greedy_flow"
    assert result["diagnostics"]["prediction_feasible"] is False


def test_backend_discovery_tool():
    tools = DecisionFlowTools(model=decision_model, workflow=workflow_document())
    result = tools.call("decisionflow_backends", {})

    assert set(result) == {
        "a_star",
        "beam_search",
        "dynamic_programming",
        "enumeration",
        "greedy",
        "pc",
        "sampling",
        "sat",
        "top_k",
    }
    assert result["greedy"]["capabilities"] == [
        "point_prediction",
        "feasibility_check",
    ]
