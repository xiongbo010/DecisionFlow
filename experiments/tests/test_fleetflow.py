from decisionflow_experiments.fleetflow import (
    MyopicFleetScorer,
    build_graph,
    canonical_instance,
    initial_state,
    prune_goal_paths,
    run_instance,
    transition,
)


def test_fleetflow_counterfactual_transition_rules():
    instance = canonical_instance(0)
    state = initial_state(instance)
    state = transition(instance, state, "assign_chilled_v1")
    state = transition(instance, state, "assign_heavy_v2")
    premature = transition(instance, state, "dispatch")
    assert premature.phase == "failed"
    assert premature.fault == "freight_route_missing"

    state = transition(instance, state, "charge_v1")
    state = transition(instance, state, "precool_v1")
    low_bridge = transition(instance, state, "plan_short")
    failed = transition(instance, low_bridge, "dispatch")
    assert failed.phase == "failed"
    assert failed.fault == "freight_vehicle_exceeds_route_clearance"


def test_fleetflow_exact_inference_beats_local_and_one_step_methods():
    result = run_instance(canonical_instance(3), horizon=7, scorer_name="myopic")
    assert not result["baselines"]["local_argmax"]["success"]
    assert not result["baselines"]["one_step_shield"]["success"]
    assert not result["baselines"]["constrained_beam"]["1"]["success"]
    assert result["baselines"]["best_first"]["success"]
    assert result["baselines"]["dynamic_programming"]["success"]
    assert result["decisionflow"]["success"]
    assert result["decisionflow"]["actions"][-2:] == ["dispatch", "verify"]
    assert result["decisionflow"]["valid_mass"] > 0
    assert result["exactness_check"]["map_match"]
    assert result["exactness_check"]["z_abs_error"] < 1e-12


def test_fleetflow_graph_contains_multiple_valid_orderings():
    instance = canonical_instance(5)
    graph = build_graph(instance, horizon=7)
    viable, pruned = prune_goal_paths(graph)
    scorer = MyopicFleetScorer(instance, 7)
    assert graph.initial_node in viable
    assert len(graph.terminals) > 0
    assert len(pruned[graph.initial_node]) >= 2
    row = scorer.score([(graph.initial_node, initial_state(instance), 0)])[
        graph.initial_node
    ]
    assert abs(sum(row.values()) - 1.0) < 1e-12


def test_fleetflow_exactness_with_early_terminal_states():
    result = run_instance(canonical_instance(0), horizon=9, scorer_name="myopic")
    assert result["decisionflow"]["steps"] < result["horizon"]
    assert result["exactness_check"]["map_match"]
    assert result["exactness_check"]["z_abs_error"] < 1e-12
