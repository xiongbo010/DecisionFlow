"""Finite-horizon trajectory inference for action policies.

This module deliberately separates controllable actions from stochastic
environment transitions. It performs exact dynamic programming over a finite
state graph and therefore does not condition the environment on desired
outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Union

from .backends.registry import BackendRegistry


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
    valid_mass: Optional[float]
    first_action_marginals: Mapping[str, float]
    trajectory_map: Tuple[Tuple[str, str], ...]
    map_probability: Optional[float]
    states_visited: int
    edges_evaluated: int
    inference_ms: float
    exact: bool = True
    backend: str = "dynamic_programming"
    prediction_kind: str = "trajectory_map"
    capabilities: Tuple[str, ...] = (
        "valid_mass",
        "first_action_marginals",
        "joint_map",
    )
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, query: str) -> bool:
        return query in self.capabilities

    def to_dict(self):
        return {
            "prediction": [list(item) for item in self.trajectory_map],
            "prediction_kind": self.prediction_kind,
            "valid_mass": self.valid_mass,
            "first_action_marginals": dict(self.first_action_marginals),
            "trajectory_map": [list(item) for item in self.trajectory_map],
            "map_probability": self.map_probability,
            "states_visited": self.states_visited,
            "edges_evaluated": self.edges_evaluated,
            "inference_ms": self.inference_ms,
            "exact": self.exact,
            "backend": self.backend,
            "capabilities": list(self.capabilities),
            "diagnostics": dict(self.diagnostics),
        }


class TrajectoryEngine:
    """Select and run a pluggable inference method over a trajectory model."""

    def __init__(
        self,
        backend: Union[str, Any] = "dynamic_programming",
        *,
        backend_options: Optional[Mapping[str, Any]] = None,
        backend_registry: Optional[BackendRegistry] = None,
    ) -> None:
        from .trajectory_backends import create_trajectory_backend_registry

        self.backend_registry = backend_registry or create_trajectory_backend_registry()
        self.backend_options = dict(backend_options or {})
        self.backend = self._resolve_backend(backend, self.backend_options)

    def _resolve_backend(
        self,
        backend: Union[str, Any],
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if isinstance(backend, str):
            return self.backend_registry.create(backend, **dict(options or {}))
        return backend

    def available_backends(self) -> Mapping[str, Any]:
        return self.backend_registry.describe()

    def infer(
        self,
        spec: TrajectorySpec,
        *,
        backend: Optional[Union[str, Any]] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
    ) -> TrajectoryResult:
        if spec.horizon < 0:
            raise ValueError("horizon must be non-negative")
        selected = (
            self.backend
            if backend is None
            else self._resolve_backend(backend, backend_options)
        )
        return selected.infer(spec)


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
