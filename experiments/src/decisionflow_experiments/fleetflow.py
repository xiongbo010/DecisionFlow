"""FleetFlow: a parameterized, multi-step fleet-dispatch experiment.

The task is inspired by JevAny's archived Fleet Dispatch case.  This module is
an independently executable benchmark environment: the public replay supplies
the task motif, while the transition system below supplies the counterfactual
states required for controlled comparisons.

All methods receive the same state, candidate actions, local probabilities,
and transition rules.  DecisionFlow-Core is used unchanged for exact
finite-horizon inference.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
import time
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from decisionflow.trajectory import TrajectoryEngine, TrajectorySpec, Transition

ROUTES = ("short", "clear")


@dataclass(frozen=True)
class Vehicle:
    id: str
    kind: str
    capacity_kg: int
    cold_capable: bool
    initial_range_km: int
    charged_range_km: int
    initial_temperature_c: int
    height_m: float


@dataclass(frozen=True)
class FleetInstance:
    id: str
    vehicles: tuple[Vehicle, ...]
    chilled_weight_kg: int
    chilled_distance_km: int
    chilled_min_c: int
    chilled_max_c: int
    heavy_weight_kg: int
    short_distance_km: int
    short_clearance_m: float
    clear_distance_km: int
    clear_clearance_m: float
    charge_minutes: int
    precool_minutes: int
    time_budget_minutes: int
    operation_budget: int

    @property
    def vehicle_index(self) -> dict[str, Vehicle]:
        return {vehicle.id: vehicle for vehicle in self.vehicles}


@dataclass(frozen=True)
class FleetState:
    chilled_vehicle: str | None
    heavy_vehicle: str | None
    ranges_km: tuple[int, ...]
    temperatures_c: tuple[int, ...]
    route: str | None
    elapsed_minutes: int
    operations: int
    phase: str = "planning"  # planning, delivered, verified, failed
    fault: str | None = None


@dataclass
class FleetGraph:
    initial_node: str
    states: dict[str, FleetState]
    depths: dict[str, int]
    transitions: dict[str, dict[str, str]]
    terminals: set[str]
    layers: list[set[str]]
    actions: tuple[str, ...]


def canonical_instance(seed: int = 0) -> FleetInstance:
    """Generate a feasible instance with stable roles and varied measurements."""
    rng = random.Random(seed)
    chilled_distance = rng.randint(48, 58)
    clear_distance = rng.randint(65, 75)
    vehicles = (
        Vehicle(
            "V1",
            "chilled van",
            rng.randint(95, 125),
            True,
            rng.randint(25, 40),
            max(chilled_distance + rng.randint(15, 28), 78),
            rng.randint(12, 17),
            round(rng.uniform(2.25, 2.55), 2),
        ),
        Vehicle(
            "V2",
            "cargo truck",
            rng.randint(340, 440),
            False,
            clear_distance + rng.randint(15, 35),
            clear_distance + rng.randint(15, 35),
            rng.randint(18, 23),
            round(rng.uniform(3.35, 3.75), 2),
        ),
        Vehicle(
            "V3",
            "city cargo bike",
            rng.randint(20, 38),
            False,
            rng.randint(16, 28),
            rng.randint(25, 38),
            rng.randint(18, 23),
            round(rng.uniform(1.45, 1.75), 2),
        ),
    )
    return FleetInstance(
        id=f"fleet-{seed:04d}",
        vehicles=vehicles,
        chilled_weight_kg=rng.randint(55, 85),
        chilled_distance_km=chilled_distance,
        chilled_min_c=2,
        chilled_max_c=8,
        heavy_weight_kg=rng.randint(220, 315),
        short_distance_km=rng.randint(40, 48),
        short_clearance_m=round(rng.uniform(2.8, 3.1), 2),
        clear_distance_km=clear_distance,
        clear_clearance_m=round(rng.uniform(4.05, 4.45), 2),
        charge_minutes=rng.randint(17, 23),
        precool_minutes=rng.randint(9, 14),
        time_budget_minutes=45,
        operation_budget=7,
    )


def initial_state(instance: FleetInstance) -> FleetState:
    return FleetState(
        chilled_vehicle=None,
        heavy_vehicle=None,
        ranges_km=tuple(vehicle.initial_range_km for vehicle in instance.vehicles),
        temperatures_c=tuple(
            vehicle.initial_temperature_c for vehicle in instance.vehicles
        ),
        route=None,
        elapsed_minutes=0,
        operations=0,
    )


def action_catalog(instance: FleetInstance) -> dict[str, str]:
    result: dict[str, str] = {}
    for vehicle in instance.vehicles:
        result[f"assign_chilled_{vehicle.id.lower()}"] = (
            f"Assign the chilled order to {vehicle.id}, the {vehicle.kind}."
        )
        result[f"assign_heavy_{vehicle.id.lower()}"] = (
            f"Assign the heavy order to {vehicle.id}, the {vehicle.kind}."
        )
        if vehicle.charged_range_km > vehicle.initial_range_km:
            result[f"charge_{vehicle.id.lower()}"] = (
                f"Charge {vehicle.id} to {vehicle.charged_range_km} km usable range."
            )
        if vehicle.cold_capable:
            result[f"precool_{vehicle.id.lower()}"] = (
                f"Pre-cool {vehicle.id}'s compartment to 4 C."
            )
    result["plan_short"] = (
        f"Choose the {instance.short_distance_km} km route with "
        f"{instance.short_clearance_m:.2f} m clearance."
    )
    result["plan_clear"] = (
        f"Choose the {instance.clear_distance_km} km route with "
        f"{instance.clear_clearance_m:.2f} m clearance."
    )
    result["dispatch"] = "Dispatch both assigned vehicles."
    result["verify"] = "Verify the two delivery records and finish."
    return result


def available_actions(instance: FleetInstance, state: FleetState) -> tuple[str, ...]:
    if state.operations >= instance.operation_budget:
        return ()
    if state.phase == "delivered":
        return ("verify",)
    if state.phase != "planning":
        return ()
    actions: list[str] = []
    for index, vehicle in enumerate(instance.vehicles):
        suffix = vehicle.id.lower()
        if state.chilled_vehicle != vehicle.id:
            actions.append(f"assign_chilled_{suffix}")
        if state.heavy_vehicle != vehicle.id:
            actions.append(f"assign_heavy_{suffix}")
        if state.ranges_km[index] < vehicle.charged_range_km:
            actions.append(f"charge_{suffix}")
        if vehicle.cold_capable and state.temperatures_c[index] > 4:
            actions.append(f"precool_{suffix}")
    for route in ROUTES:
        if state.route != route:
            actions.append(f"plan_{route}")
    if state.chilled_vehicle is not None and state.heavy_vehicle is not None:
        actions.append("dispatch")
    return tuple(actions)


def _replace(state: FleetState, **changes: Any) -> FleetState:
    values = asdict(state)
    values.update(changes)
    return FleetState(**values)


def _dispatch_fault(instance: FleetInstance, state: FleetState) -> str | None:
    if state.chilled_vehicle == state.heavy_vehicle:
        return "orders_assigned_to_same_vehicle"
    if state.route is None:
        return "freight_route_missing"
    vehicles = instance.vehicle_index
    cold = vehicles[state.chilled_vehicle]
    heavy = vehicles[state.heavy_vehicle]
    cold_index = [item.id for item in instance.vehicles].index(cold.id)
    heavy_index = [item.id for item in instance.vehicles].index(heavy.id)
    if cold.capacity_kg < instance.chilled_weight_kg:
        return "chilled_payload_exceeds_capacity"
    if not cold.cold_capable:
        return "chilled_vehicle_has_no_cold_chain"
    temperature = state.temperatures_c[cold_index]
    if not instance.chilled_min_c <= temperature <= instance.chilled_max_c:
        return "chilled_temperature_out_of_range"
    if state.ranges_km[cold_index] < instance.chilled_distance_km:
        return "chilled_vehicle_range_insufficient"
    if heavy.capacity_kg < instance.heavy_weight_kg:
        return "heavy_payload_exceeds_capacity"
    distance = (
        instance.short_distance_km
        if state.route == "short"
        else instance.clear_distance_km
    )
    clearance = (
        instance.short_clearance_m
        if state.route == "short"
        else instance.clear_clearance_m
    )
    if heavy.height_m > clearance:
        return "freight_vehicle_exceeds_route_clearance"
    if state.ranges_km[heavy_index] < distance:
        return "freight_vehicle_range_insufficient"
    return None


def transition(instance: FleetInstance, state: FleetState, action: str) -> FleetState:
    if action not in available_actions(instance, state):
        return _replace(
            state,
            operations=state.operations + 1,
            phase="failed",
            fault="unavailable_action",
        )
    operations = state.operations + 1
    if action.startswith("assign_chilled_"):
        return _replace(
            state,
            chilled_vehicle=action.rsplit("_", 1)[1].upper(),
            operations=operations,
        )
    if action.startswith("assign_heavy_"):
        return _replace(
            state, heavy_vehicle=action.rsplit("_", 1)[1].upper(), operations=operations
        )
    if action.startswith("charge_"):
        vehicle_id = action.rsplit("_", 1)[1].upper()
        index = [item.id for item in instance.vehicles].index(vehicle_id)
        ranges = list(state.ranges_km)
        ranges[index] = instance.vehicles[index].charged_range_km
        elapsed = state.elapsed_minutes + instance.charge_minutes
        return _replace(
            state,
            ranges_km=tuple(ranges),
            elapsed_minutes=elapsed,
            operations=operations,
            phase="failed" if elapsed > instance.time_budget_minutes else state.phase,
            fault="depot_time_budget_exceeded"
            if elapsed > instance.time_budget_minutes
            else None,
        )
    if action.startswith("precool_"):
        vehicle_id = action.rsplit("_", 1)[1].upper()
        index = [item.id for item in instance.vehicles].index(vehicle_id)
        temperatures = list(state.temperatures_c)
        temperatures[index] = 4
        elapsed = state.elapsed_minutes + instance.precool_minutes
        return _replace(
            state,
            temperatures_c=tuple(temperatures),
            elapsed_minutes=elapsed,
            operations=operations,
            phase="failed" if elapsed > instance.time_budget_minutes else state.phase,
            fault="depot_time_budget_exceeded"
            if elapsed > instance.time_budget_minutes
            else None,
        )
    if action.startswith("plan_"):
        return _replace(
            state, route=action.removeprefix("plan_"), operations=operations
        )
    if action == "dispatch":
        fault = _dispatch_fault(instance, state)
        if fault:
            return _replace(state, operations=operations, phase="failed", fault=fault)
        return _replace(state, operations=operations, phase="delivered", fault=None)
    if action == "verify":
        return _replace(state, operations=operations, phase="verified", fault=None)
    raise ValueError(f"unknown FleetFlow action {action!r}")


def _state_digest(state: FleetState) -> str:
    payload = json.dumps(asdict(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def node_key(depth: int, state: FleetState) -> str:
    return f"{depth}:{_state_digest(state)}"


def build_graph(instance: FleetInstance, horizon: int | None = None) -> FleetGraph:
    horizon = instance.operation_budget if horizon is None else horizon
    start = initial_state(instance)
    initial = node_key(0, start)
    states = {initial: start}
    depths = {initial: 0}
    layers: list[set[str]] = [{initial}]
    transitions: dict[str, dict[str, str]] = {}
    terminals: set[str] = set()
    all_actions = tuple(action_catalog(instance))
    for depth in range(horizon + 1):
        if depth == len(layers):
            layers.append(set())
        next_layer: set[str] = set()
        for node in sorted(layers[depth]):
            state = states[node]
            if state.phase == "verified":
                terminals.add(node)
                continue
            if depth == horizon or state.phase == "failed":
                continue
            row: dict[str, str] = {}
            for action in available_actions(instance, state):
                successor = transition(instance, state, action)
                target = node_key(depth + 1, successor)
                states.setdefault(target, successor)
                depths[target] = depth + 1
                row[action] = target
                next_layer.add(target)
            transitions[node] = row
        if depth < horizon:
            layers.append(next_layer)
    return FleetGraph(
        initial, states, depths, transitions, terminals, layers, all_actions
    )


def prune_goal_paths(graph: FleetGraph) -> tuple[set[str], dict[str, dict[str, str]]]:
    viable = set(graph.terminals)
    pruned: dict[str, dict[str, str]] = {}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            row = {
                action: target
                for action, target in graph.transitions.get(node, {}).items()
                if target in viable
            }
            if row:
                viable.add(node)
                pruned[node] = row
    return viable, pruned


def count_goal_trajectories(
    graph: FleetGraph, pruned: Mapping[str, Mapping[str, str]]
) -> int:
    counts = {node: 1 for node in graph.terminals}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            counts[node] = sum(
                counts.get(target, 0) for target in pruned.get(node, {}).values()
            )
    return counts.get(graph.initial_node, 0)


def state_payload(
    instance: FleetInstance, state: FleetState, remaining: int
) -> dict[str, Any]:
    vehicles = []
    for index, vehicle in enumerate(instance.vehicles):
        vehicles.append(
            {
                "id": vehicle.id,
                "type": vehicle.kind,
                "capacity_kg": vehicle.capacity_kg,
                "cold_capable": vehicle.cold_capable,
                "usable_range_km": state.ranges_km[index],
                "temperature_c": state.temperatures_c[index],
                "height_m": vehicle.height_m,
            }
        )
    return {
        "task": "Complete both deliveries within the remaining operation and depot-time budgets.",
        "orders": {
            "chilled": {
                "weight_kg": instance.chilled_weight_kg,
                "trip_km": instance.chilled_distance_km,
                "temperature_c": [instance.chilled_min_c, instance.chilled_max_c],
            },
            "heavy": {"weight_kg": instance.heavy_weight_kg},
        },
        "routes": {
            "short": {
                "distance_km": instance.short_distance_km,
                "clearance_m": instance.short_clearance_m,
            },
            "clear": {
                "distance_km": instance.clear_distance_km,
                "clearance_m": instance.clear_clearance_m,
            },
        },
        "vehicles": vehicles,
        "current": {
            "chilled_vehicle": state.chilled_vehicle,
            "heavy_vehicle": state.heavy_vehicle,
            "route": state.route,
            "elapsed_depot_minutes": state.elapsed_minutes,
            "remaining_operations": remaining,
            "remaining_depot_minutes": instance.time_budget_minutes
            - state.elapsed_minutes,
            "phase": state.phase,
        },
    }


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    peak = max(scores.values())
    weights = {key: math.exp(value - peak) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


class MyopicFleetScorer:
    """A reproducible local policy with an intentional short-route preference."""

    def __init__(self, instance: FleetInstance, horizon: int) -> None:
        self.instance = instance
        self.horizon = horizon

    def score(
        self, requests: Iterable[tuple[str, FleetState, int]]
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for node, state, _ in requests:
            actions = available_actions(self.instance, state)
            scores = {action: -2.0 for action in actions}
            for action in actions:
                if action == "assign_chilled_v1":
                    scores[action] = 4.4 if state.chilled_vehicle is None else 0.2
                elif action == "assign_heavy_v2":
                    scores[action] = 4.2 if state.heavy_vehicle is None else 0.2
                elif action.startswith("assign_"):
                    scores[action] = -1.5
                elif action == "dispatch":
                    scores[action] = 4.8
                elif action == "charge_v1":
                    scores[action] = 3.8
                elif action == "precool_v1":
                    scores[action] = 3.6
                elif action == "plan_short":
                    scores[action] = 3.4
                elif action == "plan_clear":
                    scores[action] = 2.9
                elif action == "verify":
                    scores[action] = 6.0
            result[node] = _softmax(scores)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "scorer": "myopic-controlled",
            "purpose": "deterministic mechanism and regression evaluation",
        }


class JevAnyHTTPScorer:
    """Score FleetFlow states through a JevAny-compatible HTTP endpoint."""

    def __init__(
        self,
        instance: FleetInstance,
        horizon: int,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        cache_path: Path | None,
        timeout: float = 120.0,
    ) -> None:
        self.instance = instance
        self.horizon = horizon
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.cache_path = cache_path
        self.timeout = timeout
        self.cache: dict[str, dict[str, float]] = {}
        self.calls = 0
        self.score_ms = 0.0
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("metadata") == self._cache_metadata():
                self.cache = payload.get("scores", {})

    def _cache_metadata(self) -> dict[str, Any]:
        return {
            "benchmark": "FleetFlow",
            "instance": self.instance.id,
            "horizon": self.horizon,
            "model": self.model,
            "prompt_version": 1,
        }

    def _save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"metadata": self._cache_metadata(), "scores": self.cache},
                indent=2,
            ),
            encoding="utf-8",
        )

    def score(
        self, requests: Iterable[tuple[str, FleetState, int]]
    ) -> dict[str, dict[str, float]]:
        catalog = action_catalog(self.instance)
        result: dict[str, dict[str, float]] = {}
        for node, state, depth in requests:
            if node in self.cache:
                result[node] = self.cache[node]
                continue
            actions = available_actions(self.instance, state)
            if len(actions) == 1:
                row = {actions[0]: 1.0}
            else:
                payload = {
                    "model": self.model,
                    "state": state_payload(self.instance, state, self.horizon - depth),
                    "questions": {
                        "next_action": {
                            "type": "choice",
                            "instructions": (
                                "Choose the next operation most likely to complete both "
                                "deliveries within the remaining budgets."
                            ),
                            "criteria": {action: catalog[action] for action in actions},
                        }
                    },
                }
                headers = {
                    "content-type": "application/json",
                    "accept": "application/json",
                }
                if self.api_key:
                    headers["authorization"] = f"Bearer {self.api_key}"
                request = urllib.request.Request(
                    self.url,
                    data=json.dumps(payload).encode(),
                    headers=headers,
                    method="POST",
                )
                started = time.perf_counter()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                self.score_ms += 1000.0 * (time.perf_counter() - started)
                self.calls += 1
                source = body["answers"]["next_action"]["probabilities"]
                total = sum(max(0.0, float(source[action])) for action in actions)
                row = {
                    action: max(0.0, float(source[action])) / total
                    for action in actions
                }
            self.cache[node] = row
            result[node] = row
        self._save()
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "scorer": "jevany-http",
            "model": self.model,
            "endpoint": self.url,
            "cached_states": len(self.cache),
            "calls_this_run": self.calls,
            "score_ms_this_run": self.score_ms,
        }


class CLMFleetScorer:
    """Adapt the released CLM-8B Choice scorer to FleetFlow states."""

    def __init__(
        self,
        instance: FleetInstance,
        horizon: int,
        backend: Any,
        *,
        include_budget_context: bool = True,
    ) -> None:
        self.instance = instance
        self.horizon = horizon
        self.backend = backend
        self.include_budget_context = include_budget_context
        self._initial_scored_states = backend.scored_states
        self._initial_score_ms = backend.score_ms

    def score(
        self, requests: Iterable[tuple[str, FleetState, int]]
    ) -> dict[str, dict[str, float]]:
        catalog = action_catalog(self.instance)
        rows = []
        for node, state, depth in requests:
            actions = available_actions(self.instance, state)
            question = {
                "action": {
                    "type": "choice",
                    "instructions": (
                        "Choose the next operation most likely to complete both "
                        "deliveries within the remaining budgets."
                    ),
                    "criteria": {action: catalog[action] for action in actions},
                }
            }
            payload = state_payload(self.instance, state, self.horizon - depth)
            if not self.include_budget_context:
                payload["current"].pop("remaining_operations")
            rows.append(
                (
                    node,
                    payload,
                    question,
                )
            )
        return self.backend._score_requests(rows)

    def metadata(self) -> dict[str, Any]:
        metadata = self.backend.metadata()
        return {
            **metadata,
            "new_states_this_instance": (
                self.backend.scored_states - self._initial_scored_states
            ),
            "score_ms_this_instance": self.backend.score_ms - self._initial_score_ms,
            "adapter": "fleetflow-choice",
        }


class StaticFleetScorer:
    """Serve a shared precomputed probability table to every inference method."""

    def __init__(self, policy: Mapping[str, Mapping[str, float]]) -> None:
        self.policy = policy

    def score(
        self, requests: Iterable[tuple[str, FleetState, int]]
    ) -> dict[str, dict[str, float]]:
        return {node: dict(self.policy[node]) for node, _, _ in requests}


def _best_action(row: Mapping[str, float]) -> str:
    return max(row, key=lambda action: (row[action], action))


def rollout(
    instance: FleetInstance,
    horizon: int,
    scorer: Any,
    *,
    shield: bool,
) -> dict[str, Any]:
    state = initial_state(instance)
    actions: list[str] = []
    rejected: list[dict[str, str]] = []
    log_probability = 0.0
    for depth in range(horizon):
        if state.phase == "verified":
            break
        node = node_key(depth, state)
        row = scorer.score([(node, state, depth)])[node]
        ordered = sorted(row, key=lambda action: (-row[action], action))
        selected = None
        for action in ordered:
            successor = transition(instance, state, action)
            if shield and successor.phase == "failed":
                rejected.append(
                    {"action": action, "reason": successor.fault or "failed"}
                )
                continue
            selected = action
            state = successor
            break
        if selected is None:
            break
        actions.append(selected)
        log_probability += math.log(max(row[selected], 1e-300))
        if state.phase == "failed":
            break
    return {
        "success": state.phase == "verified",
        "steps": len(actions),
        "actions": actions,
        "path_probability": math.exp(log_probability),
        "fault": state.fault,
        "shield_rejections": rejected,
    }


def beam_search(
    instance: FleetInstance,
    horizon: int,
    scorer: Any,
    width: int,
) -> dict[str, Any]:
    beam: list[tuple[float, FleetState, tuple[str, ...]]] = [
        (0.0, initial_state(instance), ())
    ]
    completed: list[tuple[float, FleetState, tuple[str, ...]]] = []
    expanded = 0
    for depth in range(horizon):
        active = [
            item
            for item in beam
            if item[1].phase == "planning" or item[1].phase == "delivered"
        ]
        completed.extend(item for item in beam if item[1].phase == "verified")
        if not active:
            break
        requests = [(node_key(depth, state), state, depth) for _, state, _ in active]
        rows = scorer.score(requests)
        candidates: list[tuple[float, FleetState, tuple[str, ...]]] = []
        for log_probability, state, path in active:
            row = rows[node_key(depth, state)]
            expanded += 1
            for action, probability in row.items():
                successor = transition(instance, state, action)
                if successor.phase == "failed":
                    continue
                candidates.append(
                    (
                        log_probability + math.log(max(probability, 1e-300)),
                        successor,
                        path + (action,),
                    )
                )
        candidates.sort(key=lambda item: (-item[0], item[2]))
        beam = candidates[:width]
    completed.extend(item for item in beam if item[1].phase == "verified")
    if completed:
        best = max(completed, key=lambda item: (item[0], item[2]))
        return {
            "success": True,
            "steps": len(best[2]),
            "actions": list(best[2]),
            "path_probability": math.exp(best[0]),
            "expanded_states": expanded,
            "width": width,
        }
    best = (
        max(beam, key=lambda item: (item[0], item[2]))
        if beam
        else (0.0, initial_state(instance), ())
    )
    return {
        "success": False,
        "steps": len(best[2]),
        "actions": list(best[2]),
        "path_probability": math.exp(best[0]),
        "expanded_states": expanded,
        "width": width,
    }


def best_first_search(
    graph: FleetGraph,
    policy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Find the exact maximum-probability valid trajectory on the layered DAG."""
    started = time.perf_counter()
    queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), graph.initial_node)]
    best_cost = {graph.initial_node: 0.0}
    expanded = 0
    while queue:
        cost, path, node = heapq.heappop(queue)
        if cost > best_cost.get(node, math.inf) + 1e-15:
            continue
        if node in graph.terminals:
            return {
                "success": True,
                "steps": len(path),
                "actions": list(path),
                "path_probability": math.exp(-cost),
                "expanded_states": expanded,
                "inference_ms": 1000.0 * (time.perf_counter() - started),
            }
        expanded += 1
        for action, target in graph.transitions.get(node, {}).items():
            if graph.states[target].phase == "failed":
                continue
            probability = policy[node].get(action, 0.0)
            if probability <= 0:
                continue
            candidate = cost - math.log(probability)
            if candidate + 1e-15 < best_cost.get(target, math.inf):
                best_cost[target] = candidate
                heapq.heappush(queue, (candidate, path + (action,), target))
    return {
        "success": False,
        "steps": 0,
        "actions": [],
        "path_probability": 0.0,
        "expanded_states": expanded,
        "inference_ms": 1000.0 * (time.perf_counter() - started),
    }


def exact_oracle(
    graph: FleetGraph,
    pruned: Mapping[str, Mapping[str, str]],
    policy: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    z = {node: 1.0 for node in graph.terminals}
    best = {node: 1.0 for node in graph.terminals}
    paths: dict[str, tuple[str, ...]] = {node: () for node in graph.terminals}
    for depth in range(len(graph.layers) - 2, -1, -1):
        for node in graph.layers[depth]:
            if node in graph.terminals:
                continue
            branches = []
            for action, target in pruned.get(node, {}).items():
                probability = policy.get(node, {}).get(action, 0.0)
                branches.append(
                    (
                        probability * z.get(target, 0.0),
                        probability * best.get(target, 0.0),
                        action,
                        target,
                    )
                )
            z[node] = sum(item[0] for item in branches)
            if branches:
                chosen = max(branches, key=lambda item: (item[1], item[2]))
                best[node] = chosen[1]
                paths[node] = (chosen[2],) + paths[chosen[3]]
            else:
                best[node] = 0.0
                paths[node] = ()
    total = z[graph.initial_node]
    first = {
        action: policy[graph.initial_node].get(action, 0.0) * z.get(target, 0.0)
        for action, target in pruned[graph.initial_node].items()
    }
    return {
        "valid_mass": total,
        "first_action_marginals": {
            action: value / total for action, value in first.items()
        },
        "map_actions": list(paths[graph.initial_node]),
        "map_probability": best[graph.initial_node] / total,
    }


def run_instance(
    instance: FleetInstance,
    *,
    horizon: int | None = None,
    scorer_name: str = "myopic",
    base_url: str = "http://127.0.0.1:8008",
    model: str = "jevany-latest",
    api_key: str | None = "local",
    cache_path: Path | None = None,
    clm_backend: Any | None = None,
    clm_include_budget_context: bool = True,
    beam_widths: Sequence[int] = (1, 2, 4, 16),
) -> dict[str, Any]:
    horizon = instance.operation_budget if horizon is None else horizon
    if instance.operation_budget != horizon:
        instance = replace(instance, operation_budget=horizon)
    graph_started = time.perf_counter()
    graph = build_graph(instance, horizon)
    viable, pruned = prune_goal_paths(graph)
    graph_ms = 1000.0 * (time.perf_counter() - graph_started)
    if graph.initial_node not in viable:
        raise ValueError(
            f"instance {instance.id} has no successful flow within horizon {horizon}"
        )
    if scorer_name == "myopic":
        scorer: Any = MyopicFleetScorer(instance, horizon)
    elif scorer_name == "jevany":
        scorer = JevAnyHTTPScorer(
            instance,
            horizon,
            base_url=base_url,
            model=model,
            api_key=api_key,
            cache_path=cache_path,
        )
    elif scorer_name == "clm":
        if clm_backend is None:
            raise ValueError("CLM scoring requires a shared clm_backend")
        scorer = CLMFleetScorer(
            instance,
            horizon,
            clm_backend,
            include_budget_context=clm_include_budget_context,
        )
    else:
        raise ValueError(f"unknown scorer {scorer_name!r}")

    active_nodes = sorted(
        node
        for node, state in graph.states.items()
        if state.phase in {"planning", "delivered"}
        and available_actions(instance, state)
    )
    full_policy = scorer.score(
        (node, graph.states[node], graph.depths[node]) for node in active_nodes
    )
    shared_scorer = StaticFleetScorer(full_policy)
    viable_nonterminals = sorted(viable - graph.terminals)
    policy = {node: full_policy[node] for node in viable_nonterminals}
    exact = TrajectoryEngine().infer(
        TrajectorySpec(
            initial_state=graph.initial_node,
            actions=graph.actions,
            horizon=horizon,
            terminal_states=tuple(graph.terminals),
            transitions={
                node: {action: (Transition(target),) for action, target in row.items()}
                for node, row in pruned.items()
            },
            policy=policy,
            metadata={"benchmark": "FleetFlow", "instance": instance.id},
        )
    )
    dp_started = time.perf_counter()
    oracle = exact_oracle(graph, pruned, policy)
    dp_ms = 1000.0 * (time.perf_counter() - dp_started)
    best_first = best_first_search(graph, full_policy)
    map_actions = [action for action, _ in exact.trajectory_map]
    result = exact.to_dict()
    result["first_action_marginals"] = {
        action: probability
        for action, probability in exact.first_action_marginals.items()
        if probability > 0
    }
    result.update({"success": True, "steps": len(map_actions), "actions": map_actions})
    return {
        "benchmark": "FleetFlow",
        "provenance": {
            "inspiration": "JevAny Fleet Dispatch archived case",
            "environment": "independent parameterized reimplementation",
        },
        "comparison_protocol": {
            "shared_local_probabilities": True,
            "shared_transition_rules": True,
            "local_argmax": "highest-probability available action",
            "one_step_shield": "reject only actions whose immediate successor is failed",
            "constrained_beam": "drop failed successors and retain a fixed-width probability beam",
            "best_first": "exact maximum-probability valid path on the full layered state graph",
            "dynamic_programming": "exact task-specific backward recursion on the goal-reaching graph",
            "decisionflow": "exact sum-product and max-product over every goal-reaching flow",
        },
        "instance": asdict(instance),
        "horizon": horizon,
        "reachable_states": len(graph.states),
        "reachable_edges": sum(len(row) for row in graph.transitions.values()),
        "goal_states": len(graph.terminals),
        "goal_path_states": len(viable),
        "goal_path_edges": sum(len(row) for row in pruned.values()),
        "valid_trajectory_count": count_goal_trajectories(graph, pruned),
        "graph_construction_ms": graph_ms,
        "baselines": {
            "local_argmax": rollout(instance, horizon, shared_scorer, shield=False),
            "one_step_shield": rollout(instance, horizon, shared_scorer, shield=True),
            "constrained_beam": {
                str(width): beam_search(instance, horizon, shared_scorer, width)
                for width in beam_widths
            },
            "best_first": best_first,
            "dynamic_programming": {
                "success": bool(oracle["map_actions"]),
                "steps": len(oracle["map_actions"]),
                "actions": oracle["map_actions"],
                "valid_mass": oracle["valid_mass"],
                "map_probability": oracle["map_probability"],
                "inference_ms": dp_ms,
                "processed_states": len(viable),
            },
        },
        "decisionflow": result,
        "exactness_check": {
            "z_abs_error": abs(exact.valid_mass - oracle["valid_mass"]),
            "max_marginal_abs_error": max(
                abs(
                    exact.first_action_marginals.get(action, 0.0)
                    - oracle["first_action_marginals"].get(action, 0.0)
                )
                for action in set(exact.first_action_marginals)
                | set(oracle["first_action_marginals"])
            ),
            "map_probability_abs_error": abs(
                exact.map_probability - oracle["map_probability"]
            ),
            "map_match": map_actions == oracle["map_actions"],
        },
        "scoring": scorer.metadata(),
    }


def _rate(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(bool(value))
    return sum(values) / len(values)


def run_suite(
    *,
    seeds: Sequence[int],
    horizon: int = 7,
    scorer_name: str = "myopic",
    base_url: str = "http://127.0.0.1:8008",
    model: str = "jevany-latest",
    api_key: str | None = "local",
    cache_dir: Path | None = None,
    clm_cache_path: Path | None = None,
    clm_encoder: str = "mlx-community/Qwen3-8B-8bit",
    clm_checkpoint: Path | None = None,
    clm_head_repo: str = "Contrastive-LM/CLM-v0.1-8B",
    clm_head_filename: str = "CLM_v0.1-8B.pt",
    clm_include_budget_context: bool = True,
    beam_widths: Sequence[int] = (1, 2, 4, 16),
) -> dict[str, Any]:
    clm_backend = None
    if scorer_name == "clm":
        from .trex import CLMScorer

        clm_backend = CLMScorer(
            encoder=clm_encoder,
            cache_path=clm_cache_path,
            checkpoint=clm_checkpoint,
            head_repo=clm_head_repo,
            head_filename=clm_head_filename,
        )
    rows = []
    for seed in seeds:
        cache_path = None if cache_dir is None else cache_dir / f"fleet-{seed:04d}.json"
        rows.append(
            run_instance(
                canonical_instance(seed),
                horizon=horizon,
                scorer_name=scorer_name,
                base_url=base_url,
                model=model,
                api_key=api_key,
                cache_path=cache_path,
                clm_backend=clm_backend,
                clm_include_budget_context=clm_include_budget_context,
                beam_widths=beam_widths,
            )
        )
    aggregate = {
        "instances": len(rows),
        "local_argmax_success_rate": _rate(
            rows, ("baselines", "local_argmax", "success")
        ),
        "one_step_shield_success_rate": _rate(
            rows, ("baselines", "one_step_shield", "success")
        ),
        "decisionflow_success_rate": _rate(rows, ("decisionflow", "success")),
        "best_first_success_rate": _rate(rows, ("baselines", "best_first", "success")),
        "dynamic_programming_success_rate": _rate(
            rows, ("baselines", "dynamic_programming", "success")
        ),
        "mean_valid_mass": sum(row["decisionflow"]["valid_mass"] for row in rows)
        / len(rows),
        "mean_valid_trajectory_count": sum(
            row["valid_trajectory_count"] for row in rows
        )
        / len(rows),
        "mean_reachable_states": sum(row["reachable_states"] for row in rows)
        / len(rows),
        "mean_goal_path_states": sum(row["goal_path_states"] for row in rows)
        / len(rows),
        "mean_graph_construction_ms": sum(row["graph_construction_ms"] for row in rows)
        / len(rows),
        "mean_decisionflow_inference_ms": sum(
            row["decisionflow"]["inference_ms"] for row in rows
        )
        / len(rows),
        "mean_best_first_inference_ms": sum(
            row["baselines"]["best_first"]["inference_ms"] for row in rows
        )
        / len(rows),
        "mean_best_first_expanded_states": sum(
            row["baselines"]["best_first"]["expanded_states"] for row in rows
        )
        / len(rows),
        "mean_dynamic_programming_inference_ms": sum(
            row["baselines"]["dynamic_programming"]["inference_ms"] for row in rows
        )
        / len(rows),
        "scorer_calls_this_run": sum(
            int(row["scoring"].get("calls_this_run", 0)) for row in rows
        ),
        "scorer_ms_this_run": sum(
            float(
                row["scoring"].get(
                    "score_ms_this_instance",
                    row["scoring"].get("score_ms_this_run", 0.0),
                )
            )
            for row in rows
        ),
        "scorer_new_states_this_run": sum(
            int(
                row["scoring"].get(
                    "new_states_this_instance",
                    row["scoring"].get("new_states_this_run", 0),
                )
            )
            for row in rows
        ),
        "beam_success_rate": {
            str(width): _rate(
                rows, ("baselines", "constrained_beam", str(width), "success")
            )
            for width in beam_widths
        },
        "all_exactness_checks_pass": all(
            row["exactness_check"]["z_abs_error"] < 1e-12
            and row["exactness_check"]["max_marginal_abs_error"] < 1e-12
            and row["exactness_check"]["map_probability_abs_error"] < 1e-12
            and row["exactness_check"]["map_match"]
            for row in rows
        ),
    }
    return {"benchmark": "FleetFlow", "aggregate": aggregate, "instances": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(10)))
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument(
        "--scorer", choices=("myopic", "jevany", "clm"), default="myopic"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", default="jevany-latest")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--clm-cache", type=Path)
    parser.add_argument("--clm-encoder", default="mlx-community/Qwen3-8B-8bit")
    parser.add_argument("--clm-checkpoint", type=Path)
    parser.add_argument("--clm-head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    parser.add_argument("--clm-head-filename", default="CLM_v0.1-8B.pt")
    parser.add_argument("--clm-budget-blind", action="store_true")
    parser.add_argument("--beam-widths", type=int, nargs="+", default=(1, 2, 4, 16))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_suite(
        seeds=args.seeds,
        horizon=args.horizon,
        scorer_name=args.scorer,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        cache_dir=args.cache_dir,
        clm_cache_path=args.clm_cache,
        clm_encoder=args.clm_encoder,
        clm_checkpoint=args.clm_checkpoint,
        clm_head_repo=args.clm_head_repo,
        clm_head_filename=args.clm_head_filename,
        clm_include_budget_context=not args.clm_budget_blind,
        beam_widths=args.beam_widths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "aggregate": result["aggregate"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
