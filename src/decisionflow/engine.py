"""High-level frontend/backend orchestration."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

from .backends.auto import AutoBackend
from .backends.enumeration import EnumerationBackend
from .core import (
    Constraint,
    DecisionProgram,
    DecisionRequest,
    DecisionResult,
    LocalPotentials,
)
from .frontends.json import JsonFrontend, parse_constraints, parse_probabilities
from .joints import IndependentJoint, JointBuilder
from .scorers.base import Scorer


class DecisionFlow:
    """Model and query joint distributions over structured decision flows."""

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        backend: Union[str, Any] = "auto",
        joint: Optional[JointBuilder] = None,
        frontend: Optional[Any] = None,
        enumeration_limit: int = 100_000,
    ):
        self.scorer = scorer
        self.joint = joint or IndependentJoint()
        self.frontend = frontend or JsonFrontend()
        if backend == "auto":
            self.backend = AutoBackend(enumeration_limit)
        elif backend == "enumeration":
            self.backend = EnumerationBackend(enumeration_limit)
        elif backend == "sdd":
            from .backends.sdd import SDDBackend

            self.backend = SDDBackend()
        elif isinstance(backend, str):
            raise ValueError("unknown backend: %s" % backend)
        else:
            self.backend = backend

    def compile(
        self,
        request: Union[DecisionRequest, Mapping[str, Any]],
        constraints: Any = None,
    ) -> DecisionProgram:
        """Compile a typed request and constraints without invoking a scorer.

        This is the stable boundary for applications that prepare model scores
        and decision programs in separate stages.  Dataset loading, prompting,
        and benchmark-specific transformations belong to the caller.
        """
        if isinstance(request, DecisionRequest):
            if constraints is None:
                compiled_constraints = ()
            elif isinstance(constraints, Mapping) or (
                isinstance(constraints, list)
                and (not constraints or not isinstance(constraints[0], Constraint))
            ):
                compiled_constraints = parse_constraints(constraints)
            else:
                compiled_constraints = tuple(constraints)
            return DecisionProgram(request=request, constraints=compiled_constraints)
        return self.frontend.compile(request, constraints)

    def infer_program(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
    ) -> DecisionResult:
        """Infer from an already compiled program and supplied local scores."""
        normalized = potentials.normalized_for(program.request)
        return self.backend.infer(program, normalized, self.joint)

    def infer(
        self,
        request: Union[DecisionRequest, Mapping[str, Any]],
        constraints: Any = None,
        potentials: Optional[LocalPotentials] = None,
    ) -> DecisionResult:
        request_payload = request if isinstance(request, Mapping) else None
        program = self.compile(request, constraints)
        if potentials is None:
            if self.scorer is not None:
                potentials = self.scorer.score(program.request)
            elif request_payload is not None:
                potentials = parse_probabilities(request_payload, program.request)
            else:
                raise ValueError("provide a scorer or LocalPotentials")
        return self.infer_program(program, potentials)

    def infer_many(
        self,
        items: Iterable[Tuple[DecisionProgram, LocalPotentials]],
    ) -> Iterable[DecisionResult]:
        """Infer a stream of prepared programs without imposing storage policy.

        The iterable API lets research code stream large cached score files
        while the backend reuses compiled structures when their schemas match.
        Errors intentionally propagate so the caller controls experiment-level
        retry and accounting semantics.
        """
        for program, potentials in items:
            yield self.infer_program(program, potentials)
