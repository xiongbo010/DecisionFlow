"""Interchangeable inference methods over compiled structured decision flows."""

from __future__ import annotations

import heapq
import math
import random
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

from ..backends.registry import BackendRegistry
from ..errors import UnsatisfiableError
from .schema import INACTIVE, FlowResult, FlowWorld, StructuredDecisionFlow


def _flow(world: FlowWorld):
    return tuple(
        {
            "step": decision.step,
            "depth": decision.depth,
            "decisions": dict(decision.assignment),
            "transition": decision.transition,
            "next_step": decision.next_step,
        }
        for decision in world.decisions
    )


def _prediction(world: FlowWorld):
    result = {}
    for decision in world.decisions:
        result.update(decision.namespaced_assignment)
    return result


def _marginals(
    program: StructuredDecisionFlow,
    weighted_worlds: Iterable[Any],
    normalizer: float,
):
    rows: Dict[str, Dict[Any, float]] = {
        variable: {value: 0.0 for value in domain}
        for variable, domain in program.variable_domains.items()
    }
    for world, weight in weighted_worlds:
        for variable, row in rows.items():
            value = world.assignment.get(variable, INACTIVE)
            row[value] += weight
    for row in rows.values():
        for value in row:
            row[value] /= normalizer
    return rows


def _graph(program: StructuredDecisionFlow):
    outgoing = defaultdict(list)
    for edge in program.edges:
        outgoing[edge.source].append(edge)
    terminal_world = {
        world.terminal_node: world for world in program.worlds if world.terminal_node
    }
    return outgoing, terminal_world


def _point_result(
    program: StructuredDecisionFlow,
    world: FlowWorld,
    *,
    backend: str,
    exact: bool,
    capabilities: Any,
    prediction_kind: str,
    inference_ms: float,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> FlowResult:
    return FlowResult(
        prediction=_prediction(world),
        prediction_kind=prediction_kind,
        flow=_flow(world),
        marginals={},
        valid_mass=None,
        map_probability=None,
        backend=backend,
        exact=exact,
        capabilities=tuple(capabilities),
        compile_ms=program.compile_ms,
        inference_ms=inference_ms,
        diagnostics={
            **dict(program.diagnostics),
            "selected_world": world.id,
            "prediction_feasible": world.feasible,
            "terminal_reached": world.terminal,
            "hard_constraint_violations": list(world.hard_violations),
            "soft_constraint_violations": list(world.soft_violations),
            "selected_final_state": world.final_state,
            "model_calls": program.model_calls,
            "expanded_nodes": program.expanded_nodes,
            **dict(diagnostics or {}),
        },
    )


class ExactFlowBackend:
    name = "enumeration"
    exact = True
    capabilities = ("valid_mass", "marginals", "joint_map")

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        valid = [world for world in program.worlds if world.feasible]
        hard_z = sum(world.probability for world in valid)
        weighted = [(world, world.weighted_probability) for world in valid]
        normalizer = sum(weight for _, weight in weighted)
        if normalizer <= 0:
            raise UnsatisfiableError(
                "no positive-mass complete flow satisfies the policies"
            )
        best = max(valid, key=lambda world: world.weighted_probability)
        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(best),
            prediction_kind="joint_map",
            flow=_flow(best),
            marginals=_marginals(program, weighted, normalizer),
            valid_mass=hard_z,
            map_probability=best.weighted_probability / normalizer,
            backend=self.name,
            exact=True,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "valid_world_count": len(valid),
                "normalizer": normalizer,
                "selected_world": best.id,
                "selected_final_state": best.final_state,
                "selected_stop_reason": best.stop_reason,
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
            },
        )


class GreedyFlowBackend:
    name = "greedy"
    exact = False
    capabilities = ("point_prediction", "feasibility_check")

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        candidates = [world for world in program.worlds if world.locally_greedy]
        if not candidates:
            raise UnsatisfiableError("compiled workflow has no local-greedy path")
        selected = max(candidates, key=lambda world: world.probability)
        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(selected),
            prediction_kind="local_greedy_flow",
            flow=_flow(selected),
            marginals={},
            valid_mass=None,
            map_probability=None,
            backend=self.name,
            exact=False,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "selected_world": selected.id,
                "prediction_feasible": selected.feasible,
                "terminal_reached": selected.terminal,
                "hard_constraint_violations": list(selected.hard_violations),
                "soft_constraint_violations": list(selected.soft_violations),
                "stop_reason": selected.stop_reason,
                "selected_final_state": selected.final_state,
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
            },
        )


class AStarFlowBackend:
    """Exact maximum-probability feasible-flow search with a zero heuristic."""

    name = "a_star"
    exact = True
    capabilities = ("joint_map",)

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        outgoing, terminal_world = _graph(program)
        queue = [(0.0, 0, program.root)]
        sequence = 1
        expanded = 0
        generated = 1
        while queue:
            cost, _, node_id = heapq.heappop(queue)
            world = terminal_world.get(node_id)
            if world is not None:
                if not world.feasible:
                    continue
                elapsed = 1000.0 * (time.perf_counter() - started)
                return _point_result(
                    program,
                    world,
                    backend=self.name,
                    exact=True,
                    capabilities=self.capabilities,
                    prediction_kind="joint_map",
                    inference_ms=elapsed,
                    diagnostics={
                        "heuristic": "zero",
                        "expanded_prefixes": expanded,
                        "generated_prefixes": generated,
                        "map_weight": math.exp(-cost),
                    },
                )
            expanded += 1
            for edge in outgoing.get(node_id, ()):
                if edge.probability <= 0:
                    continue
                child_cost = cost - math.log(edge.probability)
                child_world = terminal_world.get(edge.target)
                if child_world is not None:
                    if not child_world.feasible or child_world.soft_weight <= 0:
                        continue
                    child_cost -= math.log(child_world.soft_weight)
                heapq.heappush(queue, (child_cost, sequence, edge.target))
                sequence += 1
                generated += 1
        raise UnsatisfiableError("A* found no feasible terminal flow")


class BeamSearchFlowBackend:
    name = "beam_search"
    exact = False
    capabilities = ("approximate_joint_map",)

    def __init__(self, width: int = 16):
        if width <= 0:
            raise ValueError("beam width must be positive")
        self.width = int(width)

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        outgoing, terminal_world = _graph(program)
        frontier = [(1.0, program.root)]
        completed = []
        expanded = 0
        generated = 1
        max_frontier = 1
        while frontier:
            candidates = []
            for probability, node_id in frontier:
                world = terminal_world.get(node_id)
                if world is not None:
                    if world.feasible:
                        completed.append((probability * world.soft_weight, world))
                    continue
                expanded += 1
                for edge in outgoing.get(node_id, ()):
                    generated += 1
                    child_probability = probability * edge.probability
                    child_world = terminal_world.get(edge.target)
                    if child_world is not None:
                        if child_world.feasible:
                            completed.append(
                                (
                                    child_probability * child_world.soft_weight,
                                    child_world,
                                )
                            )
                    else:
                        candidates.append((child_probability, edge.target))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            frontier = candidates[: self.width]
            max_frontier = max(max_frontier, len(frontier))
        if not completed:
            raise UnsatisfiableError(
                "beam search retained no feasible terminal flow; increase width"
            )
        score, selected = max(completed, key=lambda item: (item[0], item[1].id))
        elapsed = 1000.0 * (time.perf_counter() - started)
        return _point_result(
            program,
            selected,
            backend=self.name,
            exact=False,
            capabilities=self.capabilities,
            prediction_kind="beam_map",
            inference_ms=elapsed,
            diagnostics={
                "width": self.width,
                "expanded_prefixes": expanded,
                "generated_prefixes": generated,
                "max_frontier": max_frontier,
                "selected_weight": score,
            },
        )


class DynamicProgrammingFlowBackend:
    """Exact sum-product, max-product, and edge-marginal DP on the flow DAG."""

    name = "dynamic_programming"
    exact = True
    capabilities = ("valid_mass", "marginals", "joint_map")

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        outgoing, terminal_world = _graph(program)
        memo = {}

        def solve(node_id):
            if node_id in memo:
                return memo[node_id]
            world = terminal_world.get(node_id)
            if world is not None:
                if world.feasible:
                    value = (1.0, world.soft_weight, world.soft_weight, world)
                else:
                    value = (0.0, 0.0, 0.0, None)
                memo[node_id] = value
                return value
            hard_total = 0.0
            soft_total = 0.0
            best_weight = 0.0
            best_world = None
            for edge in outgoing.get(node_id, ()):
                child_hard, child_soft, child_best, child_world = solve(edge.target)
                hard_total += edge.probability * child_hard
                soft_total += edge.probability * child_soft
                candidate = edge.probability * child_best
                if candidate > best_weight:
                    best_weight = candidate
                    best_world = child_world
            value = (hard_total, soft_total, best_weight, best_world)
            memo[node_id] = value
            return value

        hard_z, normalizer, best_weight, selected = solve(program.root)
        if selected is None or normalizer <= 0:
            raise UnsatisfiableError("DP found no positive-mass feasible terminal flow")

        forward = defaultdict(float)
        forward[program.root] = 1.0
        ordered_nodes = sorted(program.nodes, key=lambda node: (node.depth, node.id))
        for node in ordered_nodes:
            for edge in outgoing.get(node.id, ()):
                forward[edge.target] += forward[node.id] * edge.probability

        marginals: Dict[str, Dict[Any, float]] = {
            variable: {value: 0.0 for value in domain}
            for variable, domain in program.variable_domains.items()
        }
        active_mass = defaultdict(float)
        for edge in program.edges:
            child_soft = solve(edge.target)[1]
            posterior_mass = (
                forward[edge.source] * edge.probability * child_soft / normalizer
            )
            for variable, value in edge.decision.namespaced_assignment.items():
                marginals[variable][value] += posterior_mass
                active_mass[variable] += posterior_mass
        for variable, row in marginals.items():
            if INACTIVE in row:
                row[INACTIVE] = max(0.0, 1.0 - active_mass[variable])

        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(selected),
            prediction_kind="joint_map",
            flow=_flow(selected),
            marginals=marginals,
            valid_mass=hard_z,
            map_probability=best_weight / normalizer,
            backend=self.name,
            exact=True,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "normalizer": normalizer,
                "selected_world": selected.id,
                "dp_subproblems": len(memo),
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
            },
        )


def _dpll(clauses, assignment):
    assignment = dict(assignment)
    simplified = clauses
    while True:
        reduced = []
        for clause in simplified:
            satisfied = False
            remainder = []
            for literal in clause:
                variable = abs(literal)
                if variable not in assignment:
                    remainder.append(literal)
                    continue
                value = assignment[variable]
                if (literal > 0 and value) or (literal < 0 and not value):
                    satisfied = True
                    break
            if satisfied:
                continue
            if not remainder:
                return None
            reduced.append(remainder)
        simplified = reduced
        if not simplified:
            return assignment
        unit = next(
            (clause[0] for clause in simplified if len(clause) == 1), None
        )
        if unit is None:
            break
        assignment[abs(unit)] = unit > 0
    variable = abs(simplified[0][0])
    for value in (True, False):
        branch = dict(assignment)
        branch[variable] = value
        result = _dpll(simplified, branch)
        if result is not None:
            return result
    return None


class SATFlowBackend:
    """Exact feasibility through a CNF encoding and dependency-free DPLL."""

    name = "sat"
    exact = True
    capabilities = ("feasible_flow", "feasibility_check")

    def __init__(self, max_variables: int = 5_000):
        self.max_variables = int(max_variables)

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        worlds = tuple(program.worlds)
        if len(worlds) > self.max_variables:
            raise ValueError(
                "SAT world encoding needs %d variables, above max_variables=%d"
                % (len(worlds), self.max_variables)
            )
        variables = {world.id: index + 1 for index, world in enumerate(worlds)}
        clauses = [list(variables.values())]
        for world in worlds:
            if not world.feasible:
                clauses.append([-variables[world.id]])
        assignment = _dpll(clauses, {})
        if assignment is None:
            raise UnsatisfiableError("SAT encoding has no feasible terminal flow")
        selected = next(
            world for world in worlds if assignment.get(variables[world.id], False)
        )
        elapsed = 1000.0 * (time.perf_counter() - started)
        return _point_result(
            program,
            selected,
            backend=self.name,
            exact=True,
            capabilities=self.capabilities,
            prediction_kind="satisfying_flow",
            inference_ms=elapsed,
            diagnostics={
                "sat_variables": len(variables),
                "sat_clauses": len(clauses),
                "solver": "decisionflow-dpll",
                "probabilities_used": False,
            },
        )


class TopKFlowBackend:
    name = "top_k"
    exact = False
    capabilities = ("approximate_marginals", "approximate_joint_map")

    def __init__(self, width: int = 32):
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = int(width)

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        retained = sorted(
            program.worlds, key=lambda world: world.probability, reverse=True
        )[: self.width]
        valid = [world for world in retained if world.feasible]
        weighted = [(world, world.weighted_probability) for world in valid]
        normalizer = sum(weight for _, weight in weighted)
        if normalizer <= 0:
            raise UnsatisfiableError(
                "top-k truncation retained no positive-mass feasible flow"
            )
        selected = max(valid, key=lambda world: world.weighted_probability)
        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(selected),
            prediction_kind="top_k_map",
            flow=_flow(selected),
            marginals=_marginals(program, weighted, normalizer),
            valid_mass=None,
            map_probability=selected.weighted_probability / normalizer,
            backend=self.name,
            exact=False,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "width": self.width,
                "retained_world_count": len(retained),
                "retained_probability_mass": sum(
                    world.probability for world in retained
                ),
                "selected_world": selected.id,
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
            },
        )


class SamplingFlowBackend:
    name = "sampling"
    exact = False
    capabilities = ("estimated_valid_mass", "estimated_marginals", "observed_map")

    def __init__(self, samples: int = 10_000, seed: Optional[int] = 0):
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.samples = int(samples)
        self.seed = seed

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        started = time.perf_counter()
        rng = random.Random(self.seed)
        worlds = tuple(program.worlds)
        weights = [world.probability for world in worlds]
        sampled = rng.choices(worlds, weights=weights, k=self.samples)
        valid = [world for world in sampled if world.feasible]
        weighted = [(world, world.soft_weight) for world in valid]
        normalizer = sum(weight for _, weight in weighted)
        if normalizer <= 0:
            raise UnsatisfiableError(
                "sampling observed no positive-weight feasible flow"
            )
        observed = set(world.id for world in valid)
        selected = max(
            (world for world in worlds if world.id in observed),
            key=lambda world: world.weighted_probability,
        )
        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(selected),
            prediction_kind="observed_map",
            flow=_flow(selected),
            marginals=_marginals(program, weighted, normalizer),
            valid_mass=len(valid) / self.samples,
            map_probability=None,
            backend=self.name,
            exact=False,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "samples": self.samples,
                "accepted_samples": len(valid),
                "selected_world": selected.id,
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
            },
        )


class CircuitFlowBackend:
    """Exact SDD lowering through a categorical variable over compiled flows.

    The workflow compiler currently materializes finite worlds before lowering.
    This keeps the semantics exact and the public backend stable while a future
    structural lowering can compile workflow factors without world expansion.
    """

    name = "pc"
    exact = True
    capabilities = ("valid_mass", "marginals", "joint_map")

    def infer(self, program: StructuredDecisionFlow) -> FlowResult:
        from ..core import (
            Constraint,
            DecisionProgram,
            DecisionRequest,
            LocalPotentials,
            Question,
        )
        from ..engine import DecisionEngine
        from ..expressions import Table

        started = time.perf_counter()
        world_ids = [world.id for world in program.worlds]
        options = list(world_ids)
        if len(options) == 1:
            options.append("__dummy_world__")
        request = DecisionRequest(
            state=program.initial_state,
            questions=(Question("world", "choice", tuple(options)),),
        )
        probabilities = {world.id: world.probability for world in program.worlds}
        if len(world_ids) == 1:
            probabilities["__dummy_world__"] = 0.0
        constraints = [
            Constraint(
                name="complete-and-hard-feasible",
                expr=Table(
                    ("world",),
                    tuple((world.id,) for world in program.worlds if world.feasible),
                    allowed=True,
                ),
                hard=True,
            )
        ]
        for policy in program.workflow.constraints:
            if policy.hard:
                continue
            constraints.append(
                Constraint(
                    name=policy.name,
                    expr=Table(
                        ("world",),
                        tuple(
                            (world.id,)
                            for world in program.worlds
                            if policy.name not in world.soft_violations
                        ),
                        allowed=True,
                    ),
                    hard=False,
                    penalty=policy.penalty,
                )
            )
        decision_program = DecisionProgram(
            request=request,
            constraints=tuple(constraints),
            name=program.workflow.name,
            version=program.workflow.version,
        )
        result = DecisionEngine(backend="sdd").infer_program(
            decision_program,
            LocalPotentials({"world": probabilities}),
        )
        posterior = result.marginals["world"]
        worlds_by_id = {world.id: world for world in program.worlds}
        selected = worlds_by_id[result.joint_map["world"]]
        marginals = _marginals(
            program,
            (
                (world, posterior.get(world.id, 0.0))
                for world in program.worlds
            ),
            1.0,
        )
        elapsed = 1000.0 * (time.perf_counter() - started)
        return FlowResult(
            prediction=_prediction(selected),
            prediction_kind="joint_map",
            flow=_flow(selected),
            marginals=marginals,
            valid_mass=result.valid_mass,
            map_probability=result.map_probability,
            backend=self.name,
            exact=True,
            capabilities=self.capabilities,
            compile_ms=program.compile_ms + result.inference.compile_ms,
            inference_ms=elapsed,
            diagnostics={
                **dict(program.diagnostics),
                "lowering": "categorical-world-sdd",
                "selected_world": selected.id,
                "model_calls": program.model_calls,
                "expanded_nodes": program.expanded_nodes,
                "circuit_nodes": result.inference.circuit_nodes,
                "circuit_elements": result.inference.circuit_elements,
                "compile_cache_hit": result.diagnostics.get("compile_cache_hit"),
            },
        )
def create_flow_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(
        "enumeration",
        lambda **options: ExactFlowBackend(**options),
        capabilities=ExactFlowBackend.capabilities,
        exact=True,
        aliases=("exact", "auto"),
        description="Exact inference over all compiled decision flows.",
    )
    registry.register(
        "pc",
        lambda **options: CircuitFlowBackend(**options),
        capabilities=CircuitFlowBackend.capabilities,
        exact=True,
        aliases=("sdd", "probabilistic_circuit"),
        description="Exact SDD inference over compiled finite decision flows.",
    )
    registry.register(
        "greedy",
        lambda **options: GreedyFlowBackend(**options),
        capabilities=GreedyFlowBackend.capabilities,
        exact=False,
        aliases=("local_argmax",),
        description="Follow the local argmax decisions through the workflow.",
    )
    registry.register(
        "a_star",
        lambda **options: AStarFlowBackend(**options),
        capabilities=AStarFlowBackend.capabilities,
        exact=True,
        aliases=("astar", "best_first", "uniform_cost"),
        description="Exact maximum-probability feasible-flow search with h=0.",
    )
    registry.register(
        "beam_search",
        lambda **options: BeamSearchFlowBackend(**options),
        capabilities=BeamSearchFlowBackend.capabilities,
        exact=False,
        aliases=("beam",),
        description="Finite-width prefix search over the compiled workflow graph.",
    )
    registry.register(
        "dynamic_programming",
        lambda **options: DynamicProgrammingFlowBackend(**options),
        capabilities=DynamicProgrammingFlowBackend.capabilities,
        exact=True,
        aliases=("dp", "sum_product"),
        description="Exact sum-product and max-product on the workflow DAG.",
    )
    registry.register(
        "sat",
        lambda **options: SATFlowBackend(**options),
        capabilities=SATFlowBackend.capabilities,
        exact=True,
        aliases=("dpll",),
        description="Exact feasible-flow selection through CNF and DPLL.",
    )
    registry.register(
        "top_k",
        lambda **options: TopKFlowBackend(**options),
        capabilities=TopKFlowBackend.capabilities,
        exact=False,
        aliases=("truncated",),
        description="Approximate inference over the highest-mass compiled flows.",
    )
    registry.register(
        "sampling",
        lambda **options: SamplingFlowBackend(**options),
        capabilities=SamplingFlowBackend.capabilities,
        exact=False,
        aliases=("monte_carlo",),
        description="Monte Carlo queries over compiled decision flows.",
    )
    return registry


class FlowInferenceEngine:
    def __init__(
        self,
        backend: str = "auto",
        backend_options: Optional[Mapping[str, Any]] = None,
        registry: Optional[BackendRegistry] = None,
    ):
        self.registry = registry or create_flow_backend_registry()
        self.backend_name = backend
        self.backend_options = dict(backend_options or {})
        self.backend = self.registry.create(backend, **self.backend_options)

    def available_backends(self):
        return self.registry.describe()

    def infer(
        self,
        program: StructuredDecisionFlow,
        backend: Optional[str] = None,
        backend_options: Optional[Mapping[str, Any]] = None,
    ) -> FlowResult:
        selected = (
            self.backend
            if backend is None
            else self.registry.create(backend, **dict(backend_options or {}))
        )
        return selected.infer(program)
