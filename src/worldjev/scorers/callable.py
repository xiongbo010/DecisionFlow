from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping

from ..core import DecisionRequest, LocalPotentials


class CallableScorer:
    """Adapt any Python callable to the WorldJev scorer protocol."""

    def __init__(self, function: Callable[[DecisionRequest], Any]):
        self.function = function

    def score(self, request: DecisionRequest) -> LocalPotentials:
        result = self.function(request)
        if inspect.isawaitable(result):
            raise TypeError("async scorers must expose a synchronous wrapper in WorldJev v0.0")
        if isinstance(result, LocalPotentials):
            return result.normalized_for(request)
        if isinstance(result, Mapping):
            return LocalPotentials(result).normalized_for(request)
        raise TypeError("scorer callable must return LocalPotentials or a mapping")

