from __future__ import annotations

import math

from ..core import DecisionProgram, LocalPotentials
from ..errors import BackendUnavailableError
from ..joints import JointBuilder
from .enumeration import EnumerationBackend


class AutoBackend:
    name = "auto"
    exact = True
    capabilities = ("valid_mass", "marginals", "joint_map")

    def __init__(self, enumeration_limit: int = 100_000):
        self.enumeration_limit = enumeration_limit
        self._sdd = None

    def infer(
        self, program: DecisionProgram, potentials: LocalPotentials, joint: JointBuilder
    ):
        worlds = math.prod(
            len(question.options) for question in program.request.questions
        )
        if worlds <= self.enumeration_limit:
            return EnumerationBackend(self.enumeration_limit).infer(
                program, potentials, joint
            )
        try:
            from .sdd import SDDBackend

            if self._sdd is None:
                self._sdd = SDDBackend()
            return self._sdd.infer(program, potentials, joint)
        except ImportError as error:
            raise BackendUnavailableError(
                "%d worlds exceed the enumeration limit and PySDD is unavailable; "
                "install decisionflow[sdd] or select a custom backend" % worlds
            ) from error
