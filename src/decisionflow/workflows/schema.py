"""Backend-neutral intermediate representation for structured decision flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from ..core import Constraint, Question
from ..expressions import Expr


INACTIVE = "__inactive__"


@dataclass(frozen=True)
class WorkflowTransition:
    target: str
    condition: Expr
    updates: Mapping[str, Any] = field(default_factory=dict)
    name: str = ""
    otherwise: bool = False


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    questions: Tuple[Question, ...] = ()
    transitions: Tuple[WorkflowTransition, ...] = ()
    terminal: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    version: str
    start: str
    steps: Mapping[str, WorkflowStep]
    constraints: Tuple[Constraint, ...] = ()
    max_steps: int = 8
    queries: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowDecision:
    step: str
    depth: int
    assignment: Mapping[str, Any]
    namespaced_assignment: Mapping[str, Any]
    probability: float
    local_greedy: bool
    transition: Optional[str]
    next_step: Optional[str]


@dataclass(frozen=True)
class FlowNode:
    id: str
    step: str
    depth: int
    state: Any
    terminal: bool = False
    stop_reason: Optional[str] = None


@dataclass(frozen=True)
class FlowEdge:
    id: str
    source: str
    target: str
    decision: FlowDecision
    probability: float


@dataclass(frozen=True)
class FlowWorld:
    id: str
    decisions: Tuple[FlowDecision, ...]
    assignment: Mapping[str, Any]
    probability: float
    terminal: bool
    final_step: str
    final_state: Any
    hard_violations: Tuple[str, ...] = ()
    soft_violations: Tuple[str, ...] = ()
    soft_weight: float = 1.0
    stop_reason: str = "terminal"
    terminal_node: str = ""

    @property
    def feasible(self) -> bool:
        return self.terminal and not self.hard_violations

    @property
    def weighted_probability(self) -> float:
        return self.probability * self.soft_weight

    @property
    def locally_greedy(self) -> bool:
        return all(decision.local_greedy for decision in self.decisions)


@dataclass(frozen=True)
class StructuredDecisionFlow:
    workflow: WorkflowSpec
    initial_state: Any
    worlds: Tuple[FlowWorld, ...]
    model_calls: int
    expanded_nodes: int
    compile_ms: float
    variable_domains: Mapping[str, Tuple[Any, ...]]
    root: str
    nodes: Tuple[FlowNode, ...]
    edges: Tuple[FlowEdge, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowResult:
    prediction: Mapping[str, Any]
    prediction_kind: str
    flow: Tuple[Mapping[str, Any], ...]
    marginals: Mapping[str, Mapping[Any, float]]
    valid_mass: Optional[float]
    map_probability: Optional[float]
    backend: str
    exact: bool
    capabilities: Tuple[str, ...]
    compile_ms: float
    inference_ms: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, query: str) -> bool:
        return query in self.capabilities

    def to_dict(self):
        return {
            "prediction": dict(self.prediction),
            "prediction_kind": self.prediction_kind,
            "flow": [dict(item) for item in self.flow],
            "marginals": {
                variable: dict(distribution)
                for variable, distribution in self.marginals.items()
            },
            "valid_mass": self.valid_mass,
            "map_probability": self.map_probability,
            "inference": {
                "backend": self.backend,
                "exact": self.exact,
                "capabilities": list(self.capabilities),
                "compile_ms": self.compile_ms,
                "inference_ms": self.inference_ms,
            },
            "diagnostics": dict(self.diagnostics),
        }
