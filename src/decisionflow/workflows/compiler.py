"""Compile a declarative workflow into explicit structured decision worlds."""

from __future__ import annotations

import copy
import itertools
import math
import time
from typing import Any, Dict, Mapping, MutableMapping, Tuple

from ..core import DecisionRequest, LocalPotentials
from ..errors import UnsatisfiableError
from ..frontends.json import parse_operand
from .schema import (
    INACTIVE,
    FlowDecision,
    FlowEdge,
    FlowNode,
    FlowWorld,
    StructuredDecisionFlow,
    WorkflowSpec,
)


def _set_path(state: Any, path: str, value: Any) -> Any:
    if not isinstance(state, MutableMapping):
        raise TypeError("workflow state updates require a mapping state")
    current = state
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value
    return state


class WorkflowCompiler:
    """Ground questions, model scores, transitions, and policies for one state."""

    def __init__(self, model: Any, max_worlds: int = 100_000):
        if not hasattr(model, "score"):
            raise TypeError("decision model must implement score(DecisionRequest)")
        if max_worlds <= 0:
            raise ValueError("max_worlds must be positive")
        self.model = model
        self.max_worlds = int(max_worlds)

    def compile(self, workflow: WorkflowSpec, state: Any) -> StructuredDecisionFlow:
        started = time.perf_counter()
        worlds = []
        nodes = []
        edges = []
        model_calls = 0
        expanded_nodes = 0
        score_cache: Dict[Any, Any] = {}
        base_environment: Dict[str, Any] = {}
        for step in workflow.steps.values():
            base_environment["active.%s" % step.id] = False
            for question in step.questions:
                if question.id in {INACTIVE} or INACTIVE in question.options:
                    raise ValueError("%s is reserved for inactive workflow values" % INACTIVE)
                base_environment["%s.%s" % (step.id, question.id)] = INACTIVE

        def add_world(
            decisions: Tuple[FlowDecision, ...],
            environment: Mapping[str, Any],
            probability: float,
            terminal: bool,
            final_step: str,
            final_state: Any,
            stop_reason: str,
            terminal_node: str,
        ) -> None:
            if len(worlds) >= self.max_worlds:
                raise ValueError(
                    "workflow compilation exceeded max_worlds=%d" % self.max_worlds
                )
            hard_violations = []
            soft_violations = []
            soft_weight = 1.0
            for constraint in workflow.constraints:
                satisfied = constraint.expr.evaluate(environment, final_state)
                if satisfied:
                    continue
                if constraint.hard:
                    hard_violations.append(constraint.name)
                else:
                    soft_violations.append(constraint.name)
                    soft_weight *= math.exp(-constraint.penalty)
            worlds.append(
                FlowWorld(
                    id="world-%d" % len(worlds),
                    decisions=decisions,
                    assignment=dict(environment),
                    probability=probability,
                    terminal=terminal,
                    final_step=final_step,
                    final_state=copy.deepcopy(final_state),
                    hard_violations=tuple(hard_violations),
                    soft_violations=tuple(soft_violations),
                    soft_weight=soft_weight,
                    stop_reason=stop_reason,
                    terminal_node=terminal_node,
                )
            )

        def expand(
            step_id: str,
            current_state: Any,
            environment: Mapping[str, Any],
            decisions: Tuple[FlowDecision, ...],
            probability: float,
            depth: int,
        ) -> str:
            nonlocal model_calls, expanded_nodes
            step = workflow.steps[step_id]
            node_id = "node-%d" % len(nodes)
            terminal = step.terminal
            stop_reason = "terminal" if terminal else None
            if not terminal and depth >= workflow.max_steps:
                stop_reason = "max_steps"
            nodes.append(
                FlowNode(
                    id=node_id,
                    step=step_id,
                    depth=depth,
                    state=copy.deepcopy(current_state),
                    terminal=terminal,
                    stop_reason=stop_reason,
                )
            )
            if step.terminal:
                add_world(
                    decisions,
                    environment,
                    probability,
                    True,
                    step_id,
                    current_state,
                    "terminal",
                    node_id,
                )
                return node_id
            if depth >= workflow.max_steps:
                add_world(
                    decisions,
                    environment,
                    probability,
                    False,
                    step_id,
                    current_state,
                    "max_steps",
                    node_id,
                )
                return node_id
            expanded_nodes += 1
            request = DecisionRequest(
                state=copy.deepcopy(current_state),
                questions=step.questions,
                metadata={
                    "workflow": workflow.name,
                    "workflow_version": workflow.version,
                    "step": step_id,
                    "depth": depth,
                    "history": [
                        {
                            "step": item.step,
                            "assignment": dict(item.assignment),
                            "next_step": item.next_step,
                        }
                        for item in decisions
                    ],
                },
            )
            cache_key = (
                step_id,
                depth,
                repr(current_state),
                repr(tuple((item.step, tuple(item.assignment.items())) for item in decisions)),
            )
            potentials = score_cache.get(cache_key)
            if potentials is None:
                raw_potentials = self.model.score(request)
                potentials = (
                    raw_potentials
                    if isinstance(raw_potentials, LocalPotentials)
                    else LocalPotentials(raw_potentials)
                ).normalized_for(request)
                score_cache[cache_key] = potentials
                model_calls += 1
            questions = step.questions
            domains = [question.options for question in questions]
            assignments = itertools.product(*domains) if domains else [()]
            for values in assignments:
                local = dict(zip((question.id for question in questions), values))
                branch_probability = 1.0
                local_greedy = True
                namespaced = {}
                branch_environment = dict(environment)
                branch_environment["active.%s" % step_id] = True
                for question, value in zip(questions, values):
                    row = potentials.values[question.id]
                    branch_probability *= row[value]
                    greedy_value = max(question.options, key=lambda option: row[option])
                    local_greedy = local_greedy and value == greedy_value
                    occurrence = "%s@%d.%s" % (step_id, depth, question.id)
                    stable = "%s.%s" % (step_id, question.id)
                    namespaced[occurrence] = value
                    branch_environment[occurrence] = value
                    branch_environment[stable] = value
                    branch_environment[question.id] = value
                selected = None
                for transition in step.transitions:
                    if transition.condition.evaluate(
                        branch_environment, current_state
                    ):
                        selected = transition
                        break
                if selected is None:
                    decision = FlowDecision(
                        step=step_id,
                        depth=depth,
                        assignment=local,
                        namespaced_assignment=namespaced,
                        probability=branch_probability,
                        local_greedy=local_greedy,
                        transition=None,
                        next_step=None,
                    )
                    leaf_id = "node-%d" % len(nodes)
                    nodes.append(
                        FlowNode(
                            id=leaf_id,
                            step=step_id,
                            depth=depth + 1,
                            state=copy.deepcopy(current_state),
                            terminal=False,
                            stop_reason="no_transition",
                        )
                    )
                    add_world(
                        decisions + (decision,),
                        branch_environment,
                        probability * branch_probability,
                        False,
                        step_id,
                        current_state,
                        "no_transition",
                        leaf_id,
                    )
                    edges.append(
                        FlowEdge(
                            id="edge-%d" % len(edges),
                            source=node_id,
                            target=leaf_id,
                            decision=decision,
                            probability=branch_probability,
                        )
                    )
                    continue
                next_state = copy.deepcopy(current_state)
                for path, raw_value in selected.updates.items():
                    value = parse_operand(raw_value).resolve(
                        branch_environment, current_state
                    )
                    _set_path(next_state, str(path), value)
                decision = FlowDecision(
                    step=step_id,
                    depth=depth,
                    assignment=local,
                    namespaced_assignment=namespaced,
                    probability=branch_probability,
                    local_greedy=local_greedy,
                    transition=selected.name,
                    next_step=selected.target,
                )
                child_id = expand(
                    selected.target,
                    next_state,
                    branch_environment,
                    decisions + (decision,),
                    probability * branch_probability,
                    depth + 1,
                )
                edges.append(
                    FlowEdge(
                        id="edge-%d" % len(edges),
                        source=node_id,
                        target=child_id,
                        decision=decision,
                        probability=branch_probability,
                    )
                )
            return node_id

        initial_environment = dict(base_environment)
        root = expand(
            workflow.start,
            copy.deepcopy(state),
            initial_environment,
            (),
            1.0,
            0,
        )
        if not worlds:
            raise UnsatisfiableError("workflow compilation produced no decision worlds")
        variable_domains: Dict[str, Tuple[Any, ...]] = {}
        observed = {}
        for world in worlds:
            for decision in world.decisions:
                for name, value in decision.namespaced_assignment.items():
                    observed.setdefault(name, [])
                    if value not in observed[name]:
                        observed[name].append(value)
        all_variables = tuple(sorted(observed))
        for variable in all_variables:
            values = list(observed[variable])
            if any(variable not in world.assignment for world in worlds):
                values.append(INACTIVE)
            variable_domains[variable] = tuple(values)
        elapsed = 1000.0 * (time.perf_counter() - started)
        return StructuredDecisionFlow(
            workflow=workflow,
            initial_state=copy.deepcopy(state),
            worlds=tuple(worlds),
            model_calls=model_calls,
            expanded_nodes=expanded_nodes,
            compile_ms=elapsed,
            variable_domains=variable_domains,
            root=root,
            nodes=tuple(nodes),
            edges=tuple(edges),
            diagnostics={
                "world_count": len(worlds),
                "complete_world_count": sum(world.terminal for world in worlds),
                "base_probability_mass": sum(world.probability for world in worlds),
                "graph_nodes": len(nodes),
                "graph_edges": len(edges),
            },
        )
