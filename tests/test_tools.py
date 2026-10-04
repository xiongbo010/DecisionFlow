from decisionflow import DecisionFlowTools


def request(identifier="one"):
    return {
        "id": identifier,
        "state": {},
        "questions": [{"id": "action", "type": "choice", "options": ["a", "b"]}],
        "probabilities": {"action": [0.25, 0.75]},
    }


def test_tool_dispatch_and_batch():
    tools = DecisionFlowTools()
    single = tools.call("decisionflow_infer", {"request": request()})
    batch = tools.call("decisionflow_evaluate", {"requests": [request("a"), request("b")]})

    assert single["joint_map"] == {"action": "b"}
    assert len(tools.schemas) == 3
    assert batch["count"] == batch["succeeded"] == 2
    assert batch["failed"] == 0


def test_trajectory_tool():
    result = DecisionFlowTools().decisionflow_trajectory(
        {
            "initial_state": "start",
            "actions": ["inspect", "finish"],
            "horizon": 2,
            "terminal_states": ["done"],
            "policy": {
                "start": {"inspect": 0.4, "finish": 0.6},
                "inspected": {"inspect": 0.1, "finish": 0.9},
            },
            "transitions": {
                "start": {"inspect": "inspected"},
                "inspected": {"finish": "done"},
            },
        }
    )
    assert result["exact"] is True
    assert result["first_action_marginals"]["inspect"] == 1.0
