from __future__ import annotations

from ..core import DecisionRequest, LocalPotentials


class PrecomputedScorer:
    def __init__(self, potentials: LocalPotentials):
        self.potentials = potentials

    def score(self, request: DecisionRequest) -> LocalPotentials:
        return self.potentials.normalized_for(request)

