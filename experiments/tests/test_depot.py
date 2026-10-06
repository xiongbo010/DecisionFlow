from pathlib import Path

from decisionflow_experiments.depot import (
    UniformPolicyScorer,
    build_layered_graph,
    count_valid_trajectories,
    prune_goal_paths,
    validate_plan,
)
from decisionflow_experiments.strips import load_typed_strips_task


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party" / "autoplanbench" / "data" / "depot_planbench"


def _plan(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def test_depot_typed_grounding_and_official_plan():
    if not DATA.exists():
        return
    task = load_typed_strips_task(
        DATA / "domain.pddl", DATA / "orig_problems/instance-2.pddl"
    )
    assert len(task.actions) == 996
    assert len([action for action in task.actions if task.applicable(task.initial_state, action)]) == 11
    report = validate_plan(task, _plan(DATA / "orig_gold_plans/instance-2_gold_plan.txt"))
    assert report["valid"]
    assert report["success"]
    assert report["steps"] == 12


def test_depot_goal_path_graph_is_finite_and_nonempty():
    if not DATA.exists():
        return
    task = load_typed_strips_task(
        DATA / "domain.pddl", DATA / "orig_problems/instance-4.pddl"
    )
    graph = build_layered_graph(task, 5)
    viable, pruned = prune_goal_paths(graph)
    assert graph.initial_node in viable
    assert graph.terminals
    assert count_valid_trajectories(graph, pruned) > 0
    scorer = UniformPolicyScorer(task)
    initial = scorer.score([(graph.initial_node, task.initial_state, 0)])
    assert abs(sum(initial[graph.initial_node].values()) - 1.0) < 1e-12
