"""AutoPlanBench Depot: multi-step Choice decisions with temporal rules.

Every reachable planning state induces one typed Choice question over the
currently executable actions.  The local model supplies action probabilities;
DecisionFlow computes the exact probability mass and MAP trajectory of complete
goal-reaching decision flows.  Greedy and finite-width beam search consume the
same local probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decisionflow.trajectory import TrajectoryEngine, TrajectorySpec, Transition

from .strips import GroundAction, State, StripsTask, action_index, load_typed_strips_task, state_key


@dataclass
class LayeredGraph:
    initial_node: str
    states: dict[str, State]
    depths: dict[str, int]
    transitions: dict[str, dict[str, str]]
    terminals: set[str]
    layers: list[set[str]]


def node_key(depth: int, state: State) -> str:
    return f"{depth}:{state_key(state)}"


def applicable_actions(task: StripsTask, state: State) -> tuple[GroundAction, ...]:
    return tuple(action for action in task.actions if task.applicable(state, action))


def build_layered_graph(task: StripsTask, horizon: int) -> LayeredGraph:
    initial = node_key(0, task.initial_state)
    states = {initial: task.initial_state}
    depths = {initial: 0}
    layers: list[set[str]] = [{initial}]
    transitions: dict[str, dict[str, str]] = {}
    terminals: set[str] = set()
    for depth in range(horizon + 1):
        if depth == len(layers):
            layers.append(set())
        if depth == horizon:
            terminals.update(node for node in layers[depth] if task.is_goal(states[node]))
            break
        next_layer: set[str] = set()
        for node in layers[depth]:
            state = states[node]
            if task.is_goal(state):
                terminals.add(node)
                continue
            row: dict[str, str] = {}
            for action in applicable_actions(task, state):
                successor = task.apply(state, action)
                target = node_key(depth + 1, successor)
                states.setdefault(target, successor)
                depths[target] = depth + 1
                row[action.key] = target
                next_layer.add(target)
            transitions[node] = row
        layers.append(next_layer)
    return LayeredGraph(initial, states, depths, transitions, terminals, layers)


def prune_goal_paths(graph: LayeredGraph) -> tuple[set[str], dict[str, dict[str, str]]]:
    viable = set(graph.terminals)
    pruned: dict[str, dict[str, str]] = {}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            row = {
                action: target
                for action, target in graph.transitions.get(node, {}).items()
                if target in viable
            }
            if row:
                viable.add(node)
                pruned[node] = row
    return viable, pruned


def _atom_text(atom: tuple[str, ...]) -> str:
    predicate, *arguments = atom
    templates = {
        "at": "{} is at {}",
        "on": "{} is on {}",
        "in": "{} is inside {}",
        "lifting": "{} is holding {}",
        "available": "{} is available",
        "clear": "{} is clear",
    }
    template = templates.get(predicate)
    return template.format(*arguments) if template else " ".join(atom)


def action_text(action: GroundAction) -> str:
    templates = {
        "drive": "Drive {} from {} to {}.",
        "lift": "Use {} to lift {} from {} at {}.",
        "drop": "Use {} to place {} on {} at {}.",
        "load": "Use {} to load {} into {} at {}.",
        "unload": "Use {} to unload {} from {} at {}.",
    }
    return templates[action.name].format(*action.arguments)


def state_prompt(task: StripsTask, state: State, depth: int, horizon: int) -> str:
    facts = sorted(atom for atom in state if atom[0] in task.dynamic_predicates)
    goals = sorted(task.positive_goals)
    return (
        "Depot planning state. Trucks move between locations. A hoist can lift a clear "
        "crate from a surface, load a held crate into a co-located truck, unload it, "
        "and place it on a clear surface. Each hoist holds at most one crate.\n\n"
        f"Step: {depth} of {horizon}; {horizon - depth} actions remain.\n\n"
        "Current facts:\n- "
        + "\n- ".join(_atom_text(atom) for atom in facts)
        + "\n\nGoal:\n- "
        + "\n- ".join(_atom_text(atom) for atom in goals)
    )


class LayaPolicyScorer:
    """Batched, cached Laya scorer for state-specific legal action sets."""

    def __init__(
        self,
        task: StripsTask,
        *,
        horizon: int,
        model_id: str,
        device: str,
        cache_path: Path | None,
        batch_size: int,
    ) -> None:
        self.task = task
        self.horizon = horizon
        self.model_id = model_id
        self.device = device
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.actions = action_index(task)
        self.cache: dict[str, dict[str, float]] = {}
        self.agent = None
        self.score_ms = 0.0
        self.model_calls = 0
        self.cache_hits = 0
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            if metadata == self._metadata():
                self.cache = {
                    str(key): {str(a): float(p) for a, p in row.items()}
                    for key, row in payload.get("scores", {}).items()
                }

    def _metadata(self) -> dict[str, Any]:
        return {
            "benchmark": "AutoPlanBench Depot",
            "prompt_version": 1,
            "model": self.model_id,
            "horizon": self.horizon,
            "candidate_policy": "all-applicable-actions",
        }

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps({"metadata": self._metadata(), "scores": self.cache}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _normalize(row: Mapping[str, float]) -> dict[str, float]:
        total = sum(max(0.0, float(value)) for value in row.values())
        if total <= 0:
            raise ValueError("Laya returned zero probability for every action")
        return {key: max(0.0, float(value)) / total for key, value in row.items()}

    def score(
        self, requests: Iterable[tuple[str, State, int]]
    ) -> dict[str, dict[str, float]]:
        rows = list(requests)
        result: dict[str, dict[str, float]] = {}
        missing_by_schema: dict[tuple[str, ...], list[tuple[str, State, int]]] = {}
        for node, state, depth in rows:
            if node in self.cache:
                result[node] = self.cache[node]
                self.cache_hits += 1
                continue
            keys = tuple(action.key for action in applicable_actions(self.task, state))
            if not keys:
                self.cache[node] = {}
                result[node] = {}
            elif len(keys) == 1:
                self.cache[node] = {keys[0]: 1.0}
                result[node] = self.cache[node]
            else:
                missing_by_schema.setdefault(keys, []).append((node, state, depth))

        started = time.perf_counter()
        for keys, group in missing_by_schema.items():
            if self.agent is None:
                from laya import Agent

                self.agent = Agent(self.model_id, device=self.device)
            criteria = {key: action_text(self.actions[key]) for key in keys}
            questions = {
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Which executable action should be taken next so that the complete "
                        "goal can be reached within the remaining action budget?"
                    ),
                    "criteria": criteria,
                }
            }
            predictions = self.agent.predict_batch(
                [state_prompt(self.task, state, depth, self.horizon) for _, state, depth in group],
                questions,
                batch_size=self.batch_size,
                max_len=1024,
                head_max_len=512,
                sort_by_length=True,
            )
            self.model_calls += 1
            for (node, _, _), prediction in zip(group, predictions):
                row = self._normalize(prediction["answers"]["next_action"]["probabilities"])
                self.cache[node] = row
                result[node] = row
        self.score_ms += 1000.0 * (time.perf_counter() - started)
        self._save()
        return result

    def metadata(self) -> dict[str, Any]:
        result = {
            "scorer": "laya",
            "model": self.model_id,
            "device": self.device,
            "cached_rows": len(self.cache),
            "cache_hits": self.cache_hits,
            "schema_batches_this_run": self.model_calls,
            "score_ms_this_run": self.score_ms,
        }
        if self.model_id == "convaiinnovations/laya-typed-decisions":
            result["calibration_note"] = (
                "Laya clipped an out-of-range Choice temperature to 0.5; "
                "treat absolute confidence as uncalibrated."
            )
        return result


class UniformPolicyScorer:
    """Deterministic test scorer implementing the same experiment interface."""

    def __init__(self, task: StripsTask) -> None:
        self.task = task

    def score(self, requests: Iterable[tuple[str, State, int]]) -> dict[str, dict[str, float]]:
        result = {}
        for node, state, _ in requests:
            actions = applicable_actions(self.task, state)
            result[node] = {action.key: 1.0 / len(actions) for action in actions}
        return result

    def metadata(self) -> dict[str, Any]:
        return {"scorer": "uniform"}


def greedy_rollout(
    task: StripsTask, horizon: int, scorer: Any
) -> dict[str, Any]:
    state = task.initial_state
    path: list[str] = []
    log_probability = 0.0
    for depth in range(horizon):
        if task.is_goal(state):
            break
        node = node_key(depth, state)
        row = scorer.score([(node, state, depth)])[node]
        if not row:
            break
        action_key = max(row, key=lambda key: (row[key], key))
        action = action_index(task)[action_key]
        path.append(action_key)
        log_probability += math.log(max(row[action_key], 1e-300))
        state = task.apply(state, action)
    return {
        "success": task.is_goal(state),
        "steps": len(path),
        "actions": path,
        "path_probability": math.exp(log_probability),
    }


def beam_search(
    task: StripsTask, horizon: int, scorer: Any, width: int
) -> dict[str, Any]:
    # (log probability, state, action path)
    beam: list[tuple[float, State, tuple[str, ...]]] = [(0.0, task.initial_state, ())]
    completed: list[tuple[float, State, tuple[str, ...]]] = []
    expanded = 0
    for depth in range(horizon):
        active = [item for item in beam if not task.is_goal(item[1])]
        completed.extend(item for item in beam if task.is_goal(item[1]))
        if not active:
            break
        requests = [(node_key(depth, state), state, depth) for _, state, _ in active]
        scored = scorer.score(requests)
        candidates: list[tuple[float, State, tuple[str, ...]]] = []
        for log_probability, state, path in active:
            row = scored[node_key(depth, state)]
            expanded += 1
            for action_key, probability in row.items():
                action = action_index(task)[action_key]
                candidates.append(
                    (
                        log_probability + math.log(max(probability, 1e-300)),
                        task.apply(state, action),
                        path + (action_key,),
                    )
                )
        candidates.sort(key=lambda item: (-item[0], item[2]))
        beam = candidates[:width]
    completed.extend(item for item in beam if task.is_goal(item[1]))
    if completed:
        best = max(completed, key=lambda item: (item[0], item[2]))
        return {
            "success": True,
            "steps": len(best[2]),
            "actions": list(best[2]),
            "path_probability": math.exp(best[0]),
            "expanded_states": expanded,
            "width": width,
        }
    best = max(beam, key=lambda item: (item[0], item[2])) if beam else (0.0, task.initial_state, ())
    return {
        "success": False,
        "steps": len(best[2]),
        "actions": list(best[2]),
        "path_probability": math.exp(best[0]),
        "expanded_states": expanded,
        "width": width,
    }


def validate_plan(task: StripsTask, actions: Sequence[str]) -> dict[str, Any]:
    index = action_index(task)
    state = task.initial_state
    invalid_step = None
    for step, key in enumerate(actions, start=1):
        action = index.get(key)
        if action is None or not task.applicable(state, action):
            invalid_step = step
            break
        state = task.apply(state, action)
    return {
        "valid": invalid_step is None,
        "success": invalid_step is None and task.is_goal(state),
        "steps": len(actions),
        "invalid_step": invalid_step,
        "actions": list(actions),
    }


def _read_plan(path: Path) -> list[str]:
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ]


def count_valid_trajectories(
    graph: LayeredGraph, pruned: Mapping[str, Mapping[str, str]]
) -> int:
    counts = {node: 1 for node in graph.terminals}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            counts[node] = sum(counts.get(target, 0) for target in pruned.get(node, {}).values())
    return counts.get(graph.initial_node, 0)


def independent_dynamic_oracle(
    graph: LayeredGraph,
    pruned: Mapping[str, Mapping[str, str]],
    policy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Independent backward calculation used to check the library result."""
    z = {node: 1.0 for node in graph.terminals}
    best = {node: 1.0 for node in graph.terminals}
    best_path: dict[str, tuple[str, ...]] = {node: () for node in graph.terminals}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            branches = [
                (
                    policy.get(node, {}).get(action, 0.0) * z.get(target, 0.0),
                    policy.get(node, {}).get(action, 0.0) * best.get(target, 0.0),
                    action,
                    target,
                )
                for action, target in pruned.get(node, {}).items()
            ]
            z[node] = sum(item[0] for item in branches)
            if branches:
                chosen = max(branches, key=lambda item: (item[1], item[2]))
                best[node] = chosen[1]
                best_path[node] = (chosen[2],) + best_path[chosen[3]]
            else:
                best[node] = 0.0
                best_path[node] = ()
    first = {
        action: policy[graph.initial_node][action] * z.get(target, 0.0)
        for action, target in pruned[graph.initial_node].items()
    }
    total = z[graph.initial_node]
    return {
        "valid_mass": total,
        "first_action_marginals": {
            action: value / total for action, value in first.items()
        },
        "map_actions": list(best_path[graph.initial_node]),
        "map_probability": best[graph.initial_node] / total,
    }


def run_pilot(
    domain_path: Path,
    problem_path: Path,
    gold_plan_path: Path,
    *,
    horizon: int,
    model_id: str,
    device: str,
    cache_path: Path | None,
    batch_size: int = 8,
    scorer_name: str = "laya",
    beam_widths: Sequence[int] = (4, 16, 64),
) -> dict[str, Any]:
    task = load_typed_strips_task(domain_path, problem_path)
    graph_started = time.perf_counter()
    graph = build_layered_graph(task, horizon)
    viable, pruned = prune_goal_paths(graph)
    graph_ms = 1000.0 * (time.perf_counter() - graph_started)
    if graph.initial_node not in viable:
        raise ValueError(f"no goal-reaching trajectory exists within horizon {horizon}")

    if scorer_name == "laya":
        scorer: Any = LayaPolicyScorer(
            task,
            horizon=horizon,
            model_id=model_id,
            device=device,
            cache_path=cache_path,
            batch_size=batch_size,
        )
    elif scorer_name == "uniform":
        scorer = UniformPolicyScorer(task)
    else:
        raise ValueError(f"unknown scorer {scorer_name!r}")

    nonterminal_viable = sorted(node for node in viable if node not in graph.terminals)
    policy = scorer.score(
        (node, graph.states[node], graph.depths[node]) for node in nonterminal_viable
    )
    transitions = {
        node: {action: (Transition(target),) for action, target in row.items()}
        for node, row in pruned.items()
    }
    represented_actions = tuple(
        sorted({action for row in policy.values() for action in row})
    )
    exact = TrajectoryEngine().infer(
        TrajectorySpec(
            initial_state=graph.initial_node,
            actions=represented_actions,
            horizon=horizon,
            terminal_states=tuple(graph.terminals),
            transitions=transitions,
            policy=policy,
            metadata={"benchmark": "AutoPlanBench Depot", "instance": problem_path.name},
        )
    )
    oracle = independent_dynamic_oracle(graph, pruned, policy)

    gold = validate_plan(task, _read_plan(gold_plan_path))
    baselines = {
        "local_greedy": greedy_rollout(task, horizon, scorer),
        "beam_search": {
            str(width): beam_search(task, horizon, scorer, width) for width in beam_widths
        },
        "official_plan": gold,
    }
    exact_actions = [action for action, _ in exact.trajectory_map]
    exact_report = exact.to_dict()
    exact_report["first_action_marginals"] = {
        action: probability
        for action, probability in exact.first_action_marginals.items()
        if probability > 0
    }
    return {
        "benchmark": "AutoPlanBench 2.0",
        "domain": "depot_planbench",
        "problem": problem_path.name,
        "horizon": horizon,
        "candidate_policy": "all applicable grounded actions",
        "grounded_actions": len(task.actions),
        "reachable_layered_states": len(graph.states),
        "reachable_edges": sum(len(row) for row in graph.transitions.values()),
        "goal_layered_states": len(graph.terminals),
        "goal_path_states": len(viable),
        "goal_path_edges": sum(len(row) for row in pruned.values()),
        "valid_trajectory_count": count_valid_trajectories(graph, pruned),
        "graph_construction_ms": graph_ms,
        "baselines": baselines,
        "decisionflow": {
            **exact_report,
            "success": True,
            "steps": len(exact_actions),
            "actions": exact_actions,
        },
        "exactness_check": {
            "z_abs_error": abs(exact.valid_mass - oracle["valid_mass"]),
            "max_marginal_abs_error": max(
                abs(
                    exact.first_action_marginals.get(action, 0.0)
                    - oracle["first_action_marginals"].get(action, 0.0)
                )
                for action in set(exact.first_action_marginals)
                | set(oracle["first_action_marginals"])
            ),
            "map_probability_abs_error": abs(
                exact.map_probability - oracle["map_probability"]
            ),
            "map_match": exact_actions == oracle["map_actions"],
        },
        "scoring": scorer.metadata(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--instance", default="2")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scorer", choices=("laya", "uniform"), default="laya")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=(4, 16, 64))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.dataset_root / "depot_planbench"
    result = run_pilot(
        root / "domain.pddl",
        root / "orig_problems" / f"instance-{args.instance}.pddl",
        root / "orig_gold_plans" / f"instance-{args.instance}_gold_plan.txt",
        horizon=args.horizon,
        model_id=args.model,
        device=args.device,
        cache_path=args.cache,
        batch_size=args.batch_size,
        scorer_name=args.scorer,
        beam_widths=args.beam_widths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
