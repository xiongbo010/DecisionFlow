from pathlib import Path

from jevany_control_panel_multistep import (
    ACTIONS,
    INITIAL_STATE,
    State,
    assert_matches,
    circuit_inference,
    enumerate_trajectories,
    load_empirical_policy,
    run,
    transition,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/jevany-review/docs/demos/case-showcase.json"


def test_temporal_rules_and_success_path() -> None:
    state = INITIAL_STATE
    actions = (
        "press_blue",
        "hold_blue",
        "press_amber",
        "press_blue",
        "hold_green",
        "press_green",
    )
    for action in actions:
        next_state = transition(state, action)
        assert next_state is not None
        state = next_state
    assert state.ready

    assert transition(INITIAL_STATE, "hold_blue") is None
    assert transition(State(False, 48, False), "press_amber") is None
    assert transition(State(True, 0, False), "hold_green") is None
    assert transition(State(True, 28, True), "press_green") == State(True, 28, True, True)


def test_circuit_matches_complete_enumeration() -> None:
    policy, _ = load_empirical_policy(SOURCE)
    circuit = circuit_inference(policy, INITIAL_STATE, horizon=8)
    enumeration = enumerate_trajectories(policy, INITIAL_STATE, horizon=8)
    assert_matches(circuit, enumeration)
    assert circuit.z > 0
    assert circuit.valid_trajectories > 1
    assert set(circuit.first_action_marginals) == set(ACTIONS)


def test_multistep_succeeds_where_local_methods_loop() -> None:
    result = run(SOURCE, horizon=8, rollout_steps=12)
    assert not result["rollouts"]["raw_greedy"]["success"]
    assert not result["rollouts"]["myopic_constraints"]["success"]
    assert result["rollouts"]["multi_step_pcl"]["success"]
    assert [row["action"] for row in result["rollouts"]["multi_step_pcl"]["trace"]] == [
        "press_blue",
        "hold_blue",
        "press_amber",
        "press_blue",
        "hold_green",
        "press_green",
    ]
