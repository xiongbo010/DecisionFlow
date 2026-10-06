"""High-level frontend/backend orchestration."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple, Union

from .backends import BackendRegistry, create_backend_registry
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


class DecisionEngine:
    """Low-level inference over an already grounded typed decision program."""

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        backend: Union[str, Any] = "auto",
        joint: Optional[JointBuilder] = None,
        frontend: Optional[Any] = None,
        enumeration_limit: int = 100_000,
        backend_options: Optional[Mapping[str, Any]] = None,
        backend_registry: Optional[BackendRegistry] = None,
    ):
        self.scorer = scorer
        self.joint = joint or IndependentJoint()
        self.frontend = frontend or JsonFrontend()
        self.enumeration_limit = enumeration_limit
        self.backend_registry = backend_registry or create_backend_registry()
        self.backend_options = dict(backend_options or {})
        self.backend = self._resolve_backend(backend, self.backend_options)

    def _resolve_backend(
        self,
        backend: Union[str, Any],
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if not isinstance(backend, str):
            return backend
        configured = dict(options or {})
        canonical = self.backend_registry.descriptor(backend).name
        if canonical == "auto":
            configured.setdefault("enumeration_limit", self.enumeration_limit)
        elif canonical == "enumeration":
            configured.setdefault("max_worlds", self.enumeration_limit)
        return self.backend_registry.create(backend, **configured)

    def available_backends(self) -> Mapping[str, Any]:
        """Return backend capability metadata without importing optional engines."""
        return self.backend_registry.describe()

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
        backend: Optional[Union[str, Any]] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
    ) -> DecisionResult:
        """Infer from an already compiled program and supplied local scores."""
        normalized = potentials.normalized_for(program.request)
        selected = (
            self.backend
            if backend is None
            else self._resolve_backend(backend, backend_options)
        )
        return selected.infer(program, normalized, self.joint)

    def infer(
        self,
        request: Union[DecisionRequest, Mapping[str, Any]],
        constraints: Any = None,
        potentials: Optional[LocalPotentials] = None,
        backend: Optional[Union[str, Any]] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
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
        return self.infer_program(
            program,
            potentials,
            backend=backend,
            backend_options=backend_options,
        )

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
