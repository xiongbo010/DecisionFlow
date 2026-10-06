"""AutoPlanBench Logistics pilot with local OpenJev action scores.

The benchmark adapter owns PDDL parsing, grounding, prompting, and metrics.
DecisionFlow receives only a finite state graph and a model-scored action
policy, preserving the boundary between reusable inference and experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from decisionflow.trajectory import TrajectoryEngine, TrajectorySpec, Transition


Atom = tuple[str, ...]
State = frozenset[Atom]


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: tuple[str, ...]
    positive_preconditions: frozenset[Atom]
    negative_preconditions: frozenset[Atom]
    add_effects: frozenset[Atom]
    delete_effects: frozenset[Atom]


@dataclass(frozen=True, order=True)
class GroundAction:
    name: str
    arguments: tuple[str, ...]
    positive_preconditions: frozenset[Atom]
    negative_preconditions: frozenset[Atom]
    add_effects: frozenset[Atom]
    delete_effects: frozenset[Atom]

    @property
    def key(self) -> str:
        return "(%s %s)" % (self.name, " ".join(self.arguments))


@dataclass(frozen=True)
class StripsTask:
    objects: tuple[str, ...]
    initial_state: State
    positive_goals: frozenset[Atom]
    negative_goals: frozenset[Atom]
    actions: tuple[GroundAction, ...]
    dynamic_predicates: frozenset[str]

    def is_goal(self, state: State) -> bool:
        return self.positive_goals <= state and not (self.negative_goals & state)

    @staticmethod
    def applicable(state: State, action: GroundAction) -> bool:
        return action.positive_preconditions <= state and not (
            action.negative_preconditions & state
        )

    @staticmethod
    def apply(state: State, action: GroundAction) -> State:
        return frozenset((state - action.delete_effects) | action.add_effects)


def _parse_sexpr(text: str) -> list[Any]:
    text = re.sub(r";[^\n]*", "", text.lower())
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    root: list[Any] = []
    stack = [root]
    for token in tokens:
        if token == "(":
            child: list[Any] = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced closing parenthesis")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1 or len(root) != 1:
        raise ValueError("invalid PDDL S-expression")
    return root[0]


def _conjunction(expr: Any) -> tuple[frozenset[Atom], frozenset[Atom]]:
    if not isinstance(expr, list) or not expr:
        raise ValueError("expected a PDDL formula")
    rows = expr[1:] if expr[0] == "and" else [expr]
    positive: set[Atom] = set()
    negative: set[Atom] = set()
    for row in rows:
        if row[0] == "not":
            negative.add(tuple(row[1]))
        else:
            positive.add(tuple(row))
    return frozenset(positive), frozenset(negative)


def _section(tree: list[Any], name: str) -> list[Any]:
    for row in tree:
        if isinstance(row, list) and row and row[0] == name:
            return row
    raise KeyError(name)


def _parse_domain(path: Path) -> tuple[tuple[ActionSchema, ...], frozenset[str]]:
    tree = _parse_sexpr(path.read_text(encoding="utf-8"))
    schemas = []
    dynamic: set[str] = set()
    for row in tree:
        if not (isinstance(row, list) and row and row[0] == ":action"):
            continue
        fields = {row[index]: row[index + 1] for index in range(2, len(row), 2)}
        pre_pos, pre_neg = _conjunction(fields[":precondition"])
        add, delete = _conjunction(fields[":effect"])
        dynamic.update(atom[0] for atom in add | delete)
        schemas.append(
            ActionSchema(
                name=str(row[1]),
                parameters=tuple(fields[":parameters"]),
                positive_preconditions=pre_pos,
                negative_preconditions=pre_neg,
                add_effects=add,
                delete_effects=delete,
            )
        )
    return tuple(schemas), frozenset(dynamic)


def _substitute(atom: Atom, binding: Mapping[str, str]) -> Atom:
    return tuple(binding.get(token, token) for token in atom)


def load_strips_task(domain_path: Path, problem_path: Path) -> StripsTask:
    """Parse the STRIPS subset used by AutoPlanBench Logistics."""
    schemas, dynamic = _parse_domain(domain_path)
    problem = _parse_sexpr(problem_path.read_text(encoding="utf-8"))
    object_tokens = _section(problem, ":objects")[1:]
    # Logistics PlanBench is untyped PDDL. Ignore optional typed-list markers
    # so the parser remains useful for simple typed STRIPS fixtures as well.
    objects: list[str] = []
    skip_type = False
    for token in object_tokens:
        if token == "-":
            skip_type = True
            continue
        if skip_type:
            skip_type = False
            continue
        objects.append(str(token))
    initial = frozenset(tuple(atom) for atom in _section(problem, ":init")[1:])
    goal_pos, goal_neg = _conjunction(_section(problem, ":goal")[1])

    actions: list[GroundAction] = []
    for schema in schemas:
        static_positive = frozenset(
            atom for atom in schema.positive_preconditions if atom[0] not in dynamic
        )
        static_negative = frozenset(
            atom for atom in schema.negative_preconditions if atom[0] not in dynamic
        )
        for values in itertools.product(objects, repeat=len(schema.parameters)):
            binding = dict(zip(schema.parameters, values))
            grounded_static_pos = {_substitute(atom, binding) for atom in static_positive}
            grounded_static_neg = {_substitute(atom, binding) for atom in static_negative}
            if not grounded_static_pos <= initial or grounded_static_neg & initial:
                continue
            grounded_add = frozenset(
                _substitute(atom, binding) for atom in schema.add_effects
            )
            grounded_delete = frozenset(
                _substitute(atom, binding) for atom in schema.delete_effects
            )
            # Same-origin/destination moves delete and re-add the same fact.
            # They are semantic no-ops and only create arbitrary loops.
            if grounded_add == grounded_delete:
                continue
            actions.append(
                GroundAction(
                    name=schema.name,
                    arguments=tuple(values),
                    positive_preconditions=frozenset(
                        _substitute(atom, binding) for atom in schema.positive_preconditions
                    ),
                    negative_preconditions=frozenset(
                        _substitute(atom, binding) for atom in schema.negative_preconditions
                    ),
                    add_effects=grounded_add,
                    delete_effects=grounded_delete,
                )
            )
    return StripsTask(
        objects=tuple(objects),
        initial_state=initial,
        positive_goals=goal_pos,
        negative_goals=goal_neg,
        actions=tuple(sorted(actions)),
        dynamic_predicates=dynamic,
    )


def state_key(state: State) -> str:
    serialized = "|".join("(" + " ".join(atom) + ")" for atom in sorted(state))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _atom_text(atom: Atom) -> str:
    predicate, *arguments = atom
    if predicate == "at":
        return f"{arguments[0]} is at {arguments[1]}"
    if predicate == "in":
        return f"{arguments[0]} is inside {arguments[1]}"
    if predicate == "in-city":
        return f"{arguments[0]} is in {arguments[1]}"
    return " ".join(atom)


def action_text(action: GroundAction) -> str:
    args = action.arguments
    templates = {
        "load-truck": "Load %s into %s at %s.",
        "load-airplane": "Load %s into %s at %s.",
        "unload-truck": "Unload %s from %s at %s.",
        "unload-airplane": "Unload %s from %s at %s.",
        "drive-truck": "Drive %s from %s to %s in %s.",
        "fly-airplane": "Fly %s from %s to %s.",
    }
    template = templates.get(action.name)
    return template % args if template else action.key


def state_prompt(task: StripsTask, state: State) -> str:
    dynamic_facts = sorted(atom for atom in state if atom[0] in task.dynamic_predicates)
    goals = sorted(task.positive_goals)
    return (
        "Logistics planning state. Packages may be loaded only when co-located "
        "with a vehicle. Trucks move within one city; airplanes move between "
        "airports. Unloading requires the package to be inside that vehicle.\n\n"
        "Current facts:\n- "
        + "\n- ".join(_atom_text(atom) for atom in dynamic_facts)
        + "\n\nGoal:\n- "
        + "\n- ".join(_atom_text(atom) for atom in goals)
    )


def reachable_graph(
    task: StripsTask, horizon: int
) -> tuple[dict[str, State], dict[str, dict[str, str]], set[str]]:
    """Build the finite layered-reachable state graph up to ``horizon``."""
    states = {state_key(task.initial_state): task.initial_state}
    frontier = {state_key(task.initial_state)}
    transitions: dict[str, dict[str, str]] = {}
    terminals: set[str] = set()
    for _ in range(horizon + 1):
        next_frontier: set[str] = set()
        for key in frontier:
            state = states[key]
            if task.is_goal(state):
                terminals.add(key)
                continue
            by_action = transitions.setdefault(key, {})
            for action in task.actions:
                if not task.applicable(state, action):
                    continue
                successor = task.apply(state, action)
                successor_key = state_key(successor)
                states.setdefault(successor_key, successor)
                by_action[action.key] = successor_key
                next_frontier.add(successor_key)
        frontier = next_frontier
    return states, transitions, terminals


def prune_to_goal_paths(
    initial_state: str,
    transitions: Mapping[str, Mapping[str, str]],
    terminals: set[str],
    horizon: int,
) -> tuple[set[str], dict[str, dict[str, str]]]:
    """Keep states and edges that occur on a terminal path within the horizon."""
    layers: list[set[str]] = [{initial_state}]
    for _ in range(horizon):
        successors = {
            target
            for state in layers[-1]
            for target in transitions.get(state, {}).values()
        }
        layers.append(successors)
    viable: list[set[str]] = [set() for _ in layers]
    viable[-1] = layers[-1] & terminals
    for depth in range(horizon - 1, -1, -1):
        viable[depth] = layers[depth] & terminals
        for state in layers[depth]:
            if any(
                target in viable[depth + 1]
                for target in transitions.get(state, {}).values()
            ):
                viable[depth].add(state)
    retained = set().union(*viable)
    pruned: dict[str, dict[str, str]] = {}
    for depth in range(horizon):
        for state in viable[depth]:
            for action, target in transitions.get(state, {}).items():
                if target in viable[depth + 1]:
                    pruned.setdefault(state, {})[action] = target
    return retained, pruned


def _normalize(row: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in row.values())
    if total <= 0:
        raise ValueError("model returned zero action mass")
    return {key: max(0.0, float(value)) / total for key, value in row.items()}


def score_states_openjev(
    task: StripsTask,
    states: Mapping[str, State],
    *,
    model_id: str,
    device: str,
    dtype: str,
    cache_path: Path | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Score the same structural action inventory at every reachable state."""
    from openjev import Choice, LocalJev

    if len(task.actions) > 26:
        raise ValueError(
            f"OpenJev supports 26 Choice options; grounded inventory has {len(task.actions)}"
        )
    criteria = {action.key: action_text(action) for action in task.actions}
    question = Choice(
        instructions=(
            "Which action should be taken next to reach the goal? Select one "
            "action that is executable in the current state and makes useful progress."
        ),
        criteria=criteria,
    )
    prompt_fingerprint = hashlib.sha256(
        json.dumps(
            {"instructions": question.instructions, "criteria": criteria},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_metadata = {
        "model": model_id,
        "dtype": dtype,
        "prompt_sha256": prompt_fingerprint,
        "action_inventory": list(criteria),
    }
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("metadata") == cache_metadata:
            cache = dict(payload.get("scores", {}))
    engine = None
    policy: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    cache_hits = 0
    for index, (key, state) in enumerate(states.items(), start=1):
        if key in cache:
            row = cache[key]
            cache_hits += 1
        else:
            if engine is None:
                engine = LocalJev(model_id, device=device, dtype=dtype)
            answer = engine.answer_one(question, state_prompt(task, state))
            row = answer["probabilities"]
            cache[key] = row
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"metadata": cache_metadata, "scores": cache}, indent=2),
                    encoding="utf-8",
                )
        policy[key] = _normalize({str(action): float(value) for action, value in row.items()})
        print(f"scored {index}/{len(states)} states", flush=True)
    elapsed = 1000.0 * (time.perf_counter() - started)
    return policy, {
        "scorer": "openjev",
        "model": model_id,
        "device": device,
        "dtype": dtype,
        "states": len(states),
        "cache_hits": cache_hits,
        "score_ms": elapsed,
        "usage": dict(engine.usage) if engine is not None else {
            "forward_passes": 0,
            "input_tokens": 0,
            "latency_ms_total": 0.0,
        },
        "prompt_sha256": prompt_fingerprint,
    }


def score_states_laya(
    task: StripsTask,
    states: Mapping[str, State],
    *,
    model_id: str,
    device: str,
    cache_path: Path | None = None,
    batch_size: int = 4,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Score reachable states with Laya's trained non-autoregressive head."""
    from laya import Agent

    criteria = {action.key: action_text(action) for action in task.actions}
    questions = {
        "next_action": {
            "type": "choice",
            "instructions": (
                "Which action should be taken next to reach the goal? Select one "
                "action that is executable in the current state and makes useful progress."
            ),
            "criteria": criteria,
        }
    }
    prompt_fingerprint = hashlib.sha256(
        json.dumps(questions, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_metadata = {
        "scorer": "laya",
        "model": model_id,
        "prompt_sha256": prompt_fingerprint,
        "action_inventory": list(criteria),
        "max_len": 1024,
        "head_max_len": 512,
    }
    cache: dict[str, Any] = {}
    if cache_path and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("metadata") == cache_metadata:
            cache = dict(payload.get("scores", {}))

    missing = [(key, state) for key, state in states.items() if key not in cache]
    usage: dict[str, float] = {"input_tokens": 0.0, "output_tokens": 0.0}
    started = time.perf_counter()
    if missing:
        agent = Agent(model_id, device=device)
        results = agent.predict_batch(
            [state_prompt(task, state) for _, state in missing],
            questions,
            batch_size=batch_size,
            max_len=1024,
            head_max_len=512,
            sort_by_length=True,
        )
        for (key, _), result in zip(missing, results):
            cache[key] = result["answers"]["next_action"]["probabilities"]
            for name in usage:
                usage[name] += float(result.get("usage", {}).get(name, 0.0))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps({"metadata": cache_metadata, "scores": cache}, indent=2),
                encoding="utf-8",
            )
    elapsed = 1000.0 * (time.perf_counter() - started)
    policy = {
        key: _normalize({str(action): float(value) for action, value in cache[key].items()})
        for key in states
    }
    return policy, {
        "scorer": "laya",
        "model": model_id,
        "device": device,
        "states": len(states),
        "cache_hits": len(states) - len(missing),
        "score_ms": elapsed,
        "usage": usage,
        "batch_size": batch_size,
        "prompt_sha256": prompt_fingerprint,
    }


def rollout(
    task: StripsTask,
    policy: Mapping[str, Mapping[str, float]],
    horizon: int,
    *,
    mask_invalid: bool,
) -> dict[str, Any]:
    state = task.initial_state
    path: list[str] = []
    invalid = False
    for _ in range(horizon):
        if task.is_goal(state):
            break
        row = policy[state_key(state)]
        ranked = sorted(row, key=lambda action: (-row[action], action))
        selected = None
        for action_key in ranked:
            action = next(item for item in task.actions if item.key == action_key)
            if task.applicable(state, action):
                selected = action
                break
            if not mask_invalid:
                selected = action
                invalid = True
                break
        if selected is None or invalid:
            if selected is not None:
                path.append(selected.key)
            break
        path.append(selected.key)
        state = task.apply(state, selected)
    return {
        "success": task.is_goal(state),
        "invalid_action": invalid,
        "steps": len(path),
        "actions": path,
    }


def exhaustive_oracle(
    initial_state: str,
    transitions: Mapping[str, Mapping[str, str]],
    terminals: set[str],
    policy: Mapping[str, Mapping[str, float]],
    horizon: int,
) -> dict[str, Any]:
    """Enumerate the small pilot's successful trajectories independently."""
    rows: list[tuple[tuple[str, ...], float]] = []

    def visit(state: str, remaining: int, path: tuple[str, ...], mass: float) -> None:
        if state in terminals:
            rows.append((path, mass))
            return
        if remaining == 0:
            return
        for action, successor in transitions.get(state, {}).items():
            visit(successor, remaining - 1, path + (action,), mass * policy[state][action])

    visit(initial_state, horizon, (), 1.0)
    z = sum(mass for _, mass in rows)
    first = {action: 0.0 for action in policy[initial_state]}
    for path, mass in rows:
        first[path[0]] += mass
    best_path, best_mass = max(rows, key=lambda row: row[1])
    return {
        "valid_trajectory_count": len(rows),
        "valid_mass": z,
        "first_action_marginals": {
            action: mass / z if z else 0.0 for action, mass in first.items()
        },
        "map_actions": list(best_path),
        "map_probability": best_mass / z if z else 0.0,
    }


def run_pilot(
    domain_path: Path,
    problem_path: Path,
    gold_plan_path: Path,
    *,
    horizon: int,
    model_id: str,
    device: str,
    dtype: str,
    cache_path: Path | None,
    scorer: str = "openjev",
    batch_size: int = 4,
) -> dict[str, Any]:
    task = load_strips_task(domain_path, problem_path)
    states, transitions, terminals = reachable_graph(task, horizon)
    retained, goal_transitions = prune_to_goal_paths(
        state_key(task.initial_state), transitions, terminals, horizon
    )
    if state_key(task.initial_state) not in retained:
        raise ValueError(f"no goal-reaching trajectory exists within horizon {horizon}")
    # Baseline rollouts may enter legal states that cannot finish within the
    # horizon, so score every nonterminal state with outgoing transitions.
    # The exact circuit below still keeps only goal-reaching branches; omitted
    # branches retain their model mass and contribute zero to valid mass.
    scored_states = {key: states[key] for key in transitions}
    if scorer == "openjev":
        policy, scoring = score_states_openjev(
            task,
            scored_states,
            model_id=model_id,
            device=device,
            dtype=dtype,
            cache_path=cache_path,
        )
    elif scorer == "laya":
        policy, scoring = score_states_laya(
            task,
            scored_states,
            model_id=model_id,
            device=device,
            cache_path=cache_path,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"unknown scorer {scorer!r}")
    transition_spec = {
        state: {
            action: (Transition(next_state),)
            for action, next_state in by_action.items()
        }
        for state, by_action in goal_transitions.items()
    }
    exact = TrajectoryEngine().infer(
        TrajectorySpec(
            initial_state=state_key(task.initial_state),
            actions=tuple(action.key for action in task.actions),
            horizon=horizon,
            terminal_states=tuple(terminals),
            transitions=transition_spec,
            policy=policy,
            metadata={"benchmark": "AutoPlanBench Logistics"},
        )
    )
    oracle = exhaustive_oracle(
        state_key(task.initial_state), goal_transitions, terminals, policy, horizon
    )
    exact_actions = [action for action, _ in exact.trajectory_map]
    marginal_error = max(
        abs(exact.first_action_marginals[action] - oracle["first_action_marginals"][action])
        for action in exact.first_action_marginals
    )
    gold_actions = [
        line.strip().lower()
        for line in gold_plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ]
    initial_row = policy[state_key(task.initial_state)]
    invalid_mass = sum(
        initial_row[action.key]
        for action in task.actions
        if not task.applicable(task.initial_state, action)
    )
    return {
        "benchmark": "AutoPlanBench 2.0",
        "domain": "logistics_planbench",
        "problem": problem_path.name,
        "horizon": horizon,
        "grounded_action_candidates": len(task.actions),
        "reachable_states": len(states),
        "states_scored": len(scored_states),
        "goal_path_states": len(retained),
        "terminal_states": len(terminals),
        "gold_plan": gold_actions,
        "initial_invalid_probability_mass": invalid_mass,
        "raw_greedy": rollout(task, policy, horizon, mask_invalid=False),
        "valid_action_greedy": rollout(task, policy, horizon, mask_invalid=True),
        "decisionflow": exact.to_dict(),
        "enumeration_check": {
            "valid_trajectory_count": oracle["valid_trajectory_count"],
            "z_abs_error": abs(exact.valid_mass - oracle["valid_mass"]),
            "max_marginal_abs_error": marginal_error,
            "map_match": exact_actions == oracle["map_actions"],
        },
        "scoring": scoring,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--instance", default="0")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--scorer", choices=("openjev", "laya"), default="openjev")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.dataset_root / "logistics_planbench"
    result = run_pilot(
        root / "domain.pddl",
        root / "adapted_instances" / f"instance-{args.instance}.pddl",
        root / "adapted_gold_plans" / f"instance-{args.instance}_gold_plan.txt",
        horizon=args.horizon,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        cache_path=args.cache,
        scorer=args.scorer,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
