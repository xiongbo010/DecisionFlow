from pathlib import Path

from decisionflow_experiments.logistics import (
    load_strips_task,
    prune_to_goal_paths,
    reachable_graph,
    state_key,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party" / "autoplanbench" / "data" / "logistics_planbench"


def test_logistics_instance_zero_grounding_and_goal_reachability():
    if not DATA.exists():
        return
    task = load_strips_task(DATA / "domain.pddl", DATA / "adapted_instances/instance-0.pddl")
    assert len(task.actions) == 26
    states, transitions, terminals = reachable_graph(task, 5)
    assert terminals
    assert transitions
    assert not task.is_goal(task.initial_state)
    retained, pruned = prune_to_goal_paths(
        state_key(task.initial_state), transitions, terminals, 5
    )
    assert len(retained) == 8
    assert sum(len(row) for row in pruned.values()) == 9
