from __future__ import annotations

from typing import Protocol

from ..core import DecisionProgram, DecisionResult, LocalPotentials
from ..joints import JointBuilder


class InferenceBackend(Protocol):
    name: str
    exact: bool

    def infer(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
        joint: JointBuilder,
    ) -> DecisionResult:
        ...

