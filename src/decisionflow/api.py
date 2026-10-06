"""Public DecisionFlow interface."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .backends.registry import BackendRegistry
from .scorers.callable import CallableScorer
from .workflows import (
    FlowInferenceEngine,
    StructuredDecisionFlow,
    WorkflowCompiler,
    WorkflowSpec,
    load_document,
    parse_workflow,
)


class DecisionFlow:
    """Compile a declarative workflow and run structured probabilistic inference.

    Users provide a decision model, a workflow document, optional policies, and
    an inference backend. Typed questions and decision-flow structure are
    instantiated by the compiler for each input state.
    """

    def __init__(
        self,
        model: Any,
        workflow: Any,
        policies: Any = None,
        *,
        backend: str = "auto",
        backend_options: Optional[Mapping[str, Any]] = None,
        max_worlds: int = 100_000,
        backend_registry: Optional[BackendRegistry] = None,
    ):
        if hasattr(model, "score"):
            scorer = model
        elif callable(model):
            scorer = CallableScorer(model)
        else:
            raise TypeError(
                "model must be a scorer object or callable returning local probabilities"
            )
        workflow_document = (
            workflow if isinstance(workflow, WorkflowSpec) else load_document(workflow)
        )
        if isinstance(workflow_document, WorkflowSpec):
            self.workflow = workflow_document
        else:
            policy_document = (
                None
                if policies is None
                else policies
                if isinstance(policies, (Mapping, Sequence))
                and not isinstance(policies, (str, bytes))
                else load_document(policies)
            )
            self.workflow = parse_workflow(workflow_document, policy_document)
        self.model = scorer
        self.compiler = WorkflowCompiler(scorer, max_worlds=max_worlds)
        self.inference = FlowInferenceEngine(
            backend=backend,
            backend_options=backend_options,
            registry=backend_registry,
        )

    @classmethod
    def from_file(
        cls,
        workflow: Any,
        model: Any,
        policies: Any = None,
        **options: Any,
    ) -> "DecisionFlow":
        return cls(
            model=model,
            workflow=workflow,
            policies=policies,
            **options,
        )

    def compile(self, state: Any) -> StructuredDecisionFlow:
        return self.compiler.compile(self.workflow, state)

    def infer(
        self,
        state: Any,
        *,
        backend: Optional[str] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
    ):
        program = self.compile(state)
        return self.inference.infer(
            program,
            backend=backend,
            backend_options=backend_options,
        )

    run = infer

    def available_backends(self):
        return self.inference.available_backends()
