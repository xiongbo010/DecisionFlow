"""Adapter for Jev-style ``answers`` response objects."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..core import DecisionRequest, LocalPotentials
from .http import request_to_json


class TypedResponseScorer:
    """Wrap an arbitrary local SDK or hosted client returning typed answers.

    The callable receives the provider-neutral request JSON and may return
    either the response object itself or ``{"answers": ...}``. This adapter is
    intentionally vendor-neutral and covers Choice/Noul/Score response shapes.
    """

    def __init__(self, call: Callable[[Mapping[str, Any]], Mapping[str, Any]]):
        self.call = call

    def score(self, request: DecisionRequest) -> LocalPotentials:
        response = self.call(request_to_json(request))
        answers = response.get("answers", response)
        values = {}
        for question in request.questions:
            answer = answers[question.id]
            if question.type in {"noul", "boolean"}:
                positive = float(answer.get("noul", answer.get("probability_true")))
                values[question.id] = {False: 1.0 - positive, True: positive}
            else:
                values[question.id] = answer["probabilities"]
        return LocalPotentials(values, {"adapter": "typed-response"}).normalized_for(
            request
        )
