from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..core import DecisionProgram


class ConstraintFrontend(Protocol):
    """Lower an external decision/constraint representation to DecisionFlow IR."""

    name: str

    def compile(self, request: Mapping[str, Any], constraints: Any = None) -> DecisionProgram:
        ...

