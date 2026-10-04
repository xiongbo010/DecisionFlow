"""High-level frontend/backend orchestration."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

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

    def infer(
        self,
        request: Union[DecisionRequest, Mapping[str, Any]],
        constraints: Any = None,
        potentials: Optional[LocalPotentials] = None,
    ) -> DecisionResult:
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
            program = DecisionProgram(request=request, constraints=compiled_constraints)
            request_payload = None
        else:
            request_payload = request
            program = self.frontend.compile(request, constraints)
        if potentials is None:
            if self.scorer is not None:
                potentials = self.scorer.score(program.request)
            elif request_payload is not None:
                potentials = parse_probabilities(request_payload, program.request)
            else:
                raise ValueError("provide a scorer or LocalPotentials")
        potentials = potentials.normalized_for(program.request)
        return self.backend.infer(program, potentials, self.joint)
