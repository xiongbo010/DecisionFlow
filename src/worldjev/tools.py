"""JSON-in/JSON-out functions suitable for agent tool registration.

Providers differ in how tools are registered, so WorldJev exposes plain Python
callables plus provider-neutral JSON Schemas. Applications keep ownership of
the configured scorer, credentials, transport, and lifecycle.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from .engine import WorldJev
from .trajectory import TrajectoryEngine, parse_trajectory


INFER_TOOL_SCHEMA = {
    "name": "worldjev_infer",
    "description": (
        "Condition typed decision probabilities on declarative constraints and "
        "return exact marginals, joint MAP, and consistency mass."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "request": {
                "type": "object",
                "description": "State, typed questions, and optionally local probabilities.",
            },
            "constraints": {
                "type": ["object", "array", "null"],
                "description": "WorldJev JSON hard and soft constraint pack.",
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    },
}

EVALUATE_TOOL_SCHEMA = {
    "name": "worldjev_evaluate",
    "description": "Run WorldJev inference over a batch of typed decision requests.",
    "input_schema": {
        "type": "object",
        "properties": {
            "requests": {"type": "array", "items": {"type": "object"}},
            "constraints": {"type": ["object", "array", "null"]},
        },
        "required": ["requests"],
        "additionalProperties": False,
    },
}

TRAJECTORY_TOOL_SCHEMA = {
    "name": "worldjev_trajectory",
    "description": (
        "Run exact finite-horizon inference over an action policy and stochastic transitions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"spec": {"type": "object"}},
        "required": ["spec"],
        "additionalProperties": False,
    },
}


class WorldJevTools:
    """Bind a configured WorldJev engine to serializable tool functions."""

    schemas = (INFER_TOOL_SCHEMA, EVALUATE_TOOL_SCHEMA, TRAJECTORY_TOOL_SCHEMA)

    def __init__(self, engine: Optional[WorldJev] = None):
        self.engine = engine or WorldJev()
        self.trajectory_engine = TrajectoryEngine()

    def worldjev_infer(
        self,
        request: Mapping[str, Any],
        constraints: Any = None,
    ) -> Dict[str, Any]:
        return self.engine.infer(request, constraints).to_dict()

    def worldjev_evaluate(
        self,
        requests: Sequence[Mapping[str, Any]],
        constraints: Any = None,
    ) -> Dict[str, Any]:
        rows = []
        for index, request in enumerate(requests):
            identifier = request.get("id", index)
            try:
                result = self.engine.infer(request, constraints)
                rows.append({"id": identifier, "status": "ok", **result.to_dict()})
            except Exception as error:
                rows.append(
                    {
                        "id": identifier,
                        "status": "error",
                        "error": type(error).__name__,
                        "message": str(error),
                    }
                )
        return {
            "count": len(rows),
            "succeeded": sum(row["status"] == "ok" for row in rows),
            "failed": sum(row["status"] == "error" for row in rows),
            "results": rows,
        }

    def worldjev_trajectory(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        return self.trajectory_engine.infer(parse_trajectory(spec)).to_dict()

    def call(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Dispatch a provider tool call by its registered name."""
        handlers = {
            "worldjev_infer": self.worldjev_infer,
            "worldjev_evaluate": self.worldjev_evaluate,
            "worldjev_trajectory": self.worldjev_trajectory,
        }
        if name not in handlers:
            raise KeyError("unknown WorldJev tool: %s" % name)
        return handlers[name](**arguments)
