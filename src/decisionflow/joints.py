"""Base-joint interfaces.

The v0.0 exact backends implement the independent product joint. The protocol
keeps directed and learned factor joints outside the frontend/backend boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from .core import DecisionRequest, LocalPotentials


class JointBuilder(Protocol):
    name: str

    def weight(
        self,
        assignment: Mapping[str, Any],
        request: DecisionRequest,
        potentials: LocalPotentials,
    ) -> float:
        ...


class IndependentJoint:
    name = "independent"

    def weight(self, assignment, request, potentials) -> float:
        result = 1.0
        for question in request.questions:
            result *= potentials.values[question.id][assignment[question.id]]
        return result

