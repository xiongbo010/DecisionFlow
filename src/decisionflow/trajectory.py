"""Finite-horizon trajectory inference for action policies.

This module deliberately separates controllable actions from stochastic
environment transitions. It performs exact dynamic programming over a finite
state graph and therefore does not condition the environment on desired
outcomes.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from .errors import UnsatisfiableError


@dataclass(frozen=True)
class Transition:
    next_state: str
    probability: float = 1.0


@dataclass(frozen=True)
class TrajectorySpec:
    initial_state: str
    actions: Tuple[str, ...]
    horizon: int
    terminal_states: Tuple[str, ...]
    transitions: Mapping[str, Mapping[str, Tuple[Transition, ...]]]
    policy: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryResult:
    valid_mass: float
    first_action_marginals: Mapping[str, float]
    trajectory_map: Tuple[Tuple[str, str], ...]
    map_probability: float
    states_visited: int
    edges_evaluated: int
    inference_ms: float
    exact: bool = True

    def to_dict(self):
        return {
            "valid_mass": self.valid_mass,
            "first_action_marginals": dict(self.first_action_marginals),
            "trajectory_map": [list(item) for item in self.trajectory_map],
            "map_probability": self.map_probability,
            "states_visited": self.states_visited,
            "edges_evaluated": self.edges_evaluated,
            "inference_ms": self.inference_ms,
            "exact": self.exact,
        }


class TrajectoryEngine:
    """Exact finite-horizon sum/max-product over an explicit state graph."""

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        if spec.horizon < 0:
            raise ValueError("horizon must be non-negative")
        terminals = set(spec.terminal_states)
        visited = set()
        edges = set()

        def policy(state: str) -> Dict[str, float]:
            source = spec.policy.get(state, {})
            row = {action: max(0.0, float(source.get(action, 0.0))) for action in spec.actions}
            total = sum(row.values())
            if total <= 0:
                return row
            return {action: value / total for action, value in row.items()}

        @functools.lru_cache(maxsize=None)
        def solve(state: str, remaining: int):
            visited.add((state, remaining))
            if state in terminals:
                return 1.0, 1.0, (), 1
            if remaining == 0:
                return 0.0, 0.0, (), 0
            local = policy(state)
            total = 0.0
            best = 0.0
            best_path = ()
            count = 0
            for action in spec.actions:
                action_probability = local[action]
                if action_probability <= 0:
                    continue
                outcomes = spec.transitions.get(state, {}).get(action, ())
                outcome_total = sum(max(0.0, item.probability) for item in outcomes)
                if outcome_total <= 0:
                    continue
                for outcome in outcomes:
                    transition_probability = max(0.0, outcome.probability) / outcome_total
                    edges.add((state, remaining, action, outcome.next_state))
                    child_z, child_best, child_path, child_count = solve(
                        outcome.next_state, remaining - 1
                    )
                    branch = action_probability * transition_probability
                    total += branch * child_z
                    count += child_count
                    candidate = branch * child_best
                    if candidate > best:
                        best = candidate
                        best_path = ((action, outcome.next_state),) + child_path
            return total, best, best_path, count

        started = time.perf_counter()
        z, best, best_path, _ = solve(spec.initial_state, spec.horizon)
        if z <= 0:
            raise UnsatisfiableError("no terminal trajectory has positive probability")
        first_mass = {action: 0.0 for action in spec.actions}
        local = policy(spec.initial_state)
        for action in spec.actions:
            outcomes = spec.transitions.get(spec.initial_state, {}).get(action, ())
            total = sum(max(0.0, item.probability) for item in outcomes)
            if total <= 0:
                continue
            for outcome in outcomes:
                transition_probability = max(0.0, outcome.probability) / total
                child_z = solve(outcome.next_state, spec.horizon - 1)[0]
                first_mass[action] += local[action] * transition_probability * child_z
        elapsed = 1000.0 * (time.perf_counter() - started)
        return TrajectoryResult(
            valid_mass=z,
            first_action_marginals={action: value / z for action, value in first_mass.items()},
            trajectory_map=best_path,
            map_probability=best / z,
            states_visited=len(visited),
            edges_evaluated=len(edges),
            inference_ms=elapsed,
        )


def parse_trajectory(payload: Mapping[str, Any]) -> TrajectorySpec:
    transitions = {}
    for state, by_action in payload["transitions"].items():
        transitions[state] = {}
        for action, raw_outcomes in by_action.items():
            if isinstance(raw_outcomes, str):
                raw_outcomes = [{"next_state": raw_outcomes, "probability": 1.0}]
            elif isinstance(raw_outcomes, Mapping) and "next_state" in raw_outcomes:
                raw_outcomes = [raw_outcomes]
            transitions[state][action] = tuple(
                Transition(str(item["next_state"]), float(item.get("probability", 1.0)))
                for item in raw_outcomes
            )
    return TrajectorySpec(
        initial_state=str(payload["initial_state"]),
        actions=tuple(map(str, payload["actions"])),
        horizon=int(payload["horizon"]),
        terminal_states=tuple(map(str, payload["terminal_states"])),
        transitions=transitions,
        policy=payload["policy"],
        metadata=payload.get("metadata", {}),
    )
