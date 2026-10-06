from __future__ import annotations

from typing import Protocol

from ..core import DecisionRequest, LocalPotentials


class Scorer(Protocol):
    """Provider-neutral scorer interface.

    Implementations may call a hosted decision API, run a local model, or load
    cached distributions. They return local probabilities only; constraints
    and inference remain backend-independent.
    """

    def score(self, request: DecisionRequest) -> LocalPotentials: ...
