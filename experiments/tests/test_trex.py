from pathlib import Path

from decisionflow_experiments.trex import (
    UniformScorer,
    build_scenario_graph,
    collect_scenarios,
    infer_graph,
)


ROOT = Path(__file__).resolve().parents[2]
CLM = ROOT / "third_party" / "clm"


def test_trex_graph_uses_core_trajectory_engine():
    if not CLM.exists():
        return
    scenarios, planner = collect_scenarios(
        CLM,
        seeds=(0,),
        scenarios_per_seed=1,
        decision_frames=6,
        capture_distance=120,
    )
    graph = build_scenario_graph(scenarios[0], planner, horizon=4, decision_frames=6)
    assert graph.initial_node in graph.nodes
    assert graph.terminals
    assert len(graph.nodes) > 4
    policy = UniformScorer().score((graph,))
    result = infer_graph(graph, policy)
    assert 0 < result["valid_mass"] <= 1
    assert abs(sum(result["first_action_marginals"].values()) - 1) < 1e-12
    assert len(result["actions"]) == 4
