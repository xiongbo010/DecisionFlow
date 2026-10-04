"""Minimal JSON HTTP adapter for closed or remotely served decision models."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, Mapping, Optional

from ..core import DecisionRequest, LocalPotentials


def request_to_json(request: DecisionRequest) -> Dict[str, Any]:
    return {
        "state": request.state,
        "questions": [
            {
                "id": question.id,
                "type": question.type,
                "instruction": question.instruction,
                "options": list(question.options),
                "criteria": dict(question.criteria),
            }
            for question in request.questions
        ],
        "metadata": dict(request.metadata),
    }


class HttpJsonScorer:
    """Call a JSON endpoint without imposing a vendor-specific SDK.

    ``response_parser`` maps the decoded response to ``{question: {value: p}}``.
    By default the adapter accepts a top-level ``probabilities`` object or the
    object itself. Authentication headers are supplied by the caller and are
    never stored in DecisionFlow artifacts.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 60.0,
        response_parser: Optional[Callable[[Any], Mapping[str, Mapping[Any, float]]]] = None,
    ):
        self.endpoint = endpoint
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.response_parser = response_parser

    def score(self, request: DecisionRequest) -> LocalPotentials:
        body = json.dumps(request_to_json(request)).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        http_request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if self.response_parser:
            values = self.response_parser(payload)
        elif "answers" in payload:
            values = {}
            for question in request.questions:
                answer = payload["answers"][question.id]
                if question.type in {"noul", "boolean"}:
                    positive = float(answer.get("noul", answer.get("probability_true")))
                    values[question.id] = {False: 1.0 - positive, True: positive}
                else:
                    values[question.id] = answer["probabilities"]
        else:
            values = payload.get("probabilities", payload)
        return LocalPotentials(values, {"provider": self.endpoint}).normalized_for(request)
