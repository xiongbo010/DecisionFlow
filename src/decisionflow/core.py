"""Backend-neutral decision types and result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .errors import InvalidProbabilityError

Scalar = Any


def canonical_value(value: Any) -> Any:
    """Convert JSON-friendly containers to stable, hashable values."""
    if isinstance(value, list):
        return tuple(canonical_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, canonical_value(item)) for key, item in value.items()))
    return value


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    options: Tuple[Scalar, ...]
    instruction: str = ""
    criteria: Mapping[Scalar, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.type.lower()
        object.__setattr__(self, "type", kind)
        if kind not in {"choice", "noul", "score", "categorical", "boolean", "ordinal"}:
            raise ValueError("unsupported question type: %s" % self.type)
        options = tuple(canonical_value(value) for value in self.options)
        if len(options) < 2 or len(set(options)) != len(options):
            raise ValueError("question %s needs at least two unique options" % self.id)
        object.__setattr__(self, "options", options)

    def rank(self, value: Scalar) -> int:
        return self.options.index(canonical_value(value))


@dataclass(frozen=True)
class DecisionRequest:
    state: Any
    questions: Tuple[Question, ...]
    evidence: Mapping[str, Scalar] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")

    @property
    def by_id(self) -> Dict[str, Question]:
        return {question.id: question for question in self.questions}


@dataclass(frozen=True)
class Constraint:
    name: str
    expr: Any
    hard: bool = True
    penalty: float = 0.0
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hard and self.penalty < 0:
            raise ValueError("soft-constraint penalty must be non-negative")


@dataclass(frozen=True)
class DecisionProgram:
    request: DecisionRequest
    constraints: Tuple[Constraint, ...] = ()
    name: str = "anonymous"
    version: str = "0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [constraint.name for constraint in self.constraints]
        if len(names) != len(set(names)):
            raise ValueError("constraint names must be unique")

    @property
    def hard_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(item for item in self.constraints if item.hard)

    @property
    def soft_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(item for item in self.constraints if not item.hard)


@dataclass(frozen=True)
class LocalPotentials:
    values: Mapping[str, Mapping[Scalar, float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_for(self, request: DecisionRequest) -> "LocalPotentials":
        normalized: Dict[str, Dict[Scalar, float]] = {}
        for question in request.questions:
            if question.id not in self.values:
                raise InvalidProbabilityError("missing probabilities for %s" % question.id)
            source = self.values[question.id]
            row: Dict[Scalar, float] = {}
            for option in question.options:
                if option in source:
                    raw = source[option]
                elif str(option) in source:
                    raw = source[str(option)]
                elif isinstance(option, bool) and str(option).lower() in source:
                    raw = source[str(option).lower()]
                else:
                    raise InvalidProbabilityError(
                        "missing probability for %s=%r" % (question.id, option)
                    )
                value = float(raw)
                if value < 0:
                    raise InvalidProbabilityError("negative probability for %s" % question.id)
                row[option] = value
            total = sum(row.values())
            if total <= 0:
                raise InvalidProbabilityError("zero probability mass for %s" % question.id)
            normalized[question.id] = {option: value / total for option, value in row.items()}
        return LocalPotentials(normalized, self.metadata)


@dataclass(frozen=True)
class InferenceInfo:
    backend: str
    exact: bool
    scope: str
    compile_ms: float = 0.0
    inference_ms: float = 0.0
    world_count: Optional[int] = None
    valid_world_count: Optional[int] = None
    circuit_nodes: Optional[int] = None
    circuit_elements: Optional[int] = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionResult:
    marginals: Mapping[str, Mapping[Scalar, float]]
    joint_map: Mapping[str, Scalar]
    valid_mass: float
    map_probability: float
    local_potentials: Mapping[str, Mapping[Scalar, float]]
    inference: InferenceInfo
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def marginal(self, question_id: str) -> Mapping[Scalar, float]:
        return self.marginals[question_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marginals": {key: dict(value) for key, value in self.marginals.items()},
            "joint_map": dict(self.joint_map),
            "valid_mass": self.valid_mass,
            "map_probability": self.map_probability,
            "local_potentials": {
                key: dict(value) for key, value in self.local_potentials.items()
            },
            "inference": {
                "backend": self.inference.backend,
                "exact": self.inference.exact,
                "scope": self.inference.scope,
                "compile_ms": self.inference.compile_ms,
                "inference_ms": self.inference.inference_ms,
                "world_count": self.inference.world_count,
                "valid_world_count": self.inference.valid_world_count,
                "circuit_nodes": self.inference.circuit_nodes,
                "circuit_elements": self.inference.circuit_elements,
                "notes": list(self.inference.notes),
            },
            "diagnostics": dict(self.diagnostics),
        }
