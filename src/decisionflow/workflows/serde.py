"""JSON/YAML loading and validation for declarative workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ..errors import ConstraintSyntaxError
from ..expressions import Bool
from ..frontends.json import parse_constraints, parse_expr, parse_request
from .schema import WorkflowSpec, WorkflowStep, WorkflowTransition


def load_document(source: Any) -> Any:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML workflows require PyYAML; use JSON or install decisionflow[yaml]"
        ) from error
    return yaml.safe_load(text)


def _questions(rows: Any):
    if not rows:
        return ()
    return parse_request({"state": {}, "questions": rows}).questions


def _transition(payload: Mapping[str, Any], index: int) -> WorkflowTransition:
    target = payload.get("goto", payload.get("target"))
    if target is None:
        raise ConstraintSyntaxError("workflow transition needs goto")
    otherwise = bool(payload.get("otherwise", False))
    condition_payload = payload.get("when", True)
    condition = Bool(True) if otherwise else parse_expr(condition_payload)
    updates = payload.get("set", payload.get("updates", {}))
    if not isinstance(updates, Mapping):
        raise ConstraintSyntaxError("transition updates must be an object")
    return WorkflowTransition(
        target=str(target),
        condition=condition,
        updates=dict(updates),
        name=str(payload.get("name", "transition-%d" % index)),
        otherwise=otherwise,
    )


def parse_workflow(payload: Mapping[str, Any], policies: Any = None) -> WorkflowSpec:
    document = payload.get("workflow", payload)
    if not isinstance(document, Mapping):
        raise ConstraintSyntaxError("workflow document must be an object")
    raw_steps = document.get("steps")
    if not isinstance(raw_steps, Mapping) or not raw_steps:
        raise ConstraintSyntaxError("workflow.steps must be a non-empty object")
    declared_terminals = set(map(str, document.get("terminal_steps", ())))
    steps = {}
    for step_id, raw in raw_steps.items():
        if not isinstance(raw, Mapping):
            raise ConstraintSyntaxError("workflow step must be an object")
        transitions = tuple(
            _transition(item, index)
            for index, item in enumerate(raw.get("transitions", ()))
        )
        steps[str(step_id)] = WorkflowStep(
            id=str(step_id),
            questions=_questions(raw.get("questions", ())),
            transitions=transitions,
            terminal=bool(raw.get("terminal", False))
            or str(step_id) in declared_terminals,
            metadata=raw.get("metadata", {}),
        )
    start = str(document.get("start", next(iter(steps))))
    if start not in steps:
        raise ConstraintSyntaxError("workflow start step does not exist: %s" % start)
    for step in steps.values():
        if step.terminal:
            continue
        if not step.transitions:
            raise ConstraintSyntaxError(
                "non-terminal workflow step needs transitions: %s" % step.id
            )
        otherwise_count = sum(item.otherwise for item in step.transitions)
        if otherwise_count > 1:
            raise ConstraintSyntaxError(
                "workflow step has multiple otherwise transitions: %s" % step.id
            )
        for transition in step.transitions:
            if transition.target not in steps:
                raise ConstraintSyntaxError(
                    "transition target does not exist: %s" % transition.target
                )
    inline_policies = document.get(
        "constraints", document.get("policies", payload.get("constraints"))
    )
    constraints = list(parse_constraints(inline_policies))
    if policies is not None:
        constraints.extend(parse_constraints(policies))
    max_steps = int(document.get("max_steps", 8))
    if max_steps <= 0:
        raise ConstraintSyntaxError("workflow max_steps must be positive")
    return WorkflowSpec(
        name=str(document.get("name", "anonymous-workflow")),
        version=str(document.get("version", "0")),
        start=start,
        steps=steps,
        constraints=tuple(constraints),
        max_steps=max_steps,
        queries=tuple(map(str, document.get("queries", ()))),
        metadata=document.get("metadata", {}),
    )
