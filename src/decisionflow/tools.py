"""Provider-neutral tool functions for declarative decision workflows."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .api import DecisionFlow
from .workflows.inference import create_flow_backend_registry


RUN_TOOL_SCHEMA = {
    "name": "decisionflow_run",
    "description": (
        "Compile a declarative business workflow with a configured decision "
        "model and run structured probabilistic inference."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "state": {"description": "Initial business state."},
            "workflow": {
                "type": ["object", "null"],
                "description": "Workflow document; optional when preconfigured.",
            },
            "policies": {
                "type": ["object", "array", "null"],
                "description": "Optional hard and soft decision policies.",
            },
            "backend": {"type": ["string", "null"]},
            "backend_options": {"type": ["object", "null"]},
            "max_worlds": {"type": ["integer", "null"], "minimum": 1},
        },
        "required": ["state"],
        "additionalProperties": False,
    },
}

BACKENDS_TOOL_SCHEMA = {
    "name": "decisionflow_backends",
    "description": "List structured-flow inference backends and capabilities.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


class DecisionFlowTools:
    schemas = (RUN_TOOL_SCHEMA, BACKENDS_TOOL_SCHEMA)

    def __init__(
        self,
        model: Any,
        workflow: Any = None,
        policies: Any = None,
        *,
        backend: str = "auto",
        backend_options: Optional[Mapping[str, Any]] = None,
        max_worlds: int = 100_000,
    ):
        self.model = model
        self.workflow = workflow
        self.policies = policies
        self.backend = backend
        self.backend_options = dict(backend_options or {})
        self.max_worlds = max_worlds

    def decisionflow_run(
        self,
        state: Any,
        workflow: Any = None,
        policies: Any = None,
        backend: Optional[str] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
        max_worlds: Optional[int] = None,
    ) -> Dict[str, Any]:
        selected_workflow = self.workflow if workflow is None else workflow
        if selected_workflow is None:
            raise ValueError("provide a workflow in the tool call or tool configuration")
        flow = DecisionFlow(
            model=self.model,
            workflow=selected_workflow,
            policies=self.policies if policies is None else policies,
            backend=backend or self.backend,
            backend_options=(
                self.backend_options if backend_options is None else backend_options
            ),
            max_worlds=max_worlds or self.max_worlds,
        )
        return flow.infer(state).to_dict()

    def decisionflow_backends(self) -> Dict[str, Any]:
        return {
            name: descriptor.to_dict()
            for name, descriptor in create_flow_backend_registry().describe().items()
        }

    def call(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        handlers = {
            "decisionflow_run": self.decisionflow_run,
            "decisionflow_backends": self.decisionflow_backends,
        }
        if name not in handlers:
            raise KeyError("unknown DecisionFlow tool: %s" % name)
        return handlers[name](**arguments)
