#!/usr/bin/env python3
"""Exact multi-step Choice pilot on JevAny's control-panel replay.

The archived JevAny records provide a probability distribution over the seven
actions at each observed fixture state.  We average distributions that share
the same symbolic fixture state to obtain an empirical Markov policy.  A small
deterministic transition model is transcribed from the action descriptions and
``fixture_spec`` included in the records.

The constrained trajectory distribution contains every action sequence that
reaches the accepted fixture state within a finite horizon without violating a
temporal precondition.  A layered sum-product/max-product circuit computes the
valid mass, first-action marginals, and trajectory MAP.  Exhaustive sequence
enumeration is retained as an independent correctness oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


ACTIONS = (
    "press_blue",
    "hold_blue",
    "press_amber",
    "hold_amber",
    "hold_green",
    "press_green",
    "finish",
)


@dataclass(frozen=True, order=True)
class State:
    powered: bool
    pressure_kpa: int
    clamped: bool
    ready: bool = False


INITIAL_STATE = State(powered=True, pressure_kpa=48, clamped=False)
DECISIVE_STATE = State(powered=False, pressure_kpa=48, clamped=False)


def state_from_record(record: Mapping[str, object]) -> State:
    fixture = record["state"]["observed_physics"]["fixture"]  # type: ignore[index]
    return State(
        powered=bool(fixture["powered"]),
        pressure_kpa=int(fixture["pressure_kpa"]),
        clamped=bool(fixture["clamped"]),
        ready=bool(fixture["ready"]),
    )


def load_empirical_policy(path: Path) -> tuple[dict[State, dict[str, float]], dict]:
    """Average Jev probabilities for identical fault-free symbolic states."""
    payload = json.loads(path.read_text())
    accum: dict[State, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    attempts = [row for row in payload["attempts"] if row["case"] == "control_panel"]
    observations = 0
    for attempt in attempts:
        for decision in attempt["decisions"]:
            fixture = decision["state"]["observed_physics"]["fixture"]
            if fixture.get("fault") is not None or fixture.get("ready"):
                continue
            state = state_from_record(decision)
            by_action = {row["action"]: float(row["probability"]) for row in decision["options"]}
            if set(by_action) != set(ACTIONS):
                raise ValueError(f"unexpected action inventory at {state}: {sorted(by_action)}")
            for action, probability in by_action.items():
                accum[state][action].append(probability)
            observations += 1

    policy: dict[State, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for state, values in accum.items():
        means = {action: sum(rows) / len(rows) for action, rows in values.items()}
        total = sum(means.values())
        policy[state] = {action: means[action] / total for action in ACTIONS}
        counts[state_label(state)] = len(next(iter(values.values())))

    metadata = {
        "attempts": len(attempts),
        "successful_attempts": sum(bool(row["success"]) for row in attempts),
        "decisions": sum(len(row["decisions"]) for row in attempts),
        "fault_free_policy_observations": observations,
        "state_observation_counts": counts,
    }
    return policy, metadata


def state_label(state: State) -> str:
    return (
        f"power={'on' if state.powered else 'off'},"
        f"pressure={state.pressure_kpa},"
        f"clamp={'closed' if state.clamped else 'open'},"
        f"ready={str(state.ready).lower()}"
    )


def transition(state: State, action: str) -> State | None:
    """Apply a hard-safe control-panel transition; ``None`` means violation.

    Rules are taken from the replay's candidate descriptions and fixture spec:
    bleeding requires isolated power; clamping requires isolated power and at
    most 5 kPa; pumping requires power and a closed clamp and may not exceed the
    35 kPa limit; acceptance requires 20--30 kPa, power, and a closed clamp.
    Opening a loaded clamp and finishing before acceptance are excluded.
    """
    if state.ready:
        return state
    if action == "press_blue":
        return State(not state.powered, state.pressure_kpa, state.clamped)
    if action == "hold_blue":
        if state.powered:
            return None
        return State(False, 0, state.clamped)
    if action == "press_amber":
        if state.powered or state.pressure_kpa > 5:
            return None
        return State(False, state.pressure_kpa, True)
    if action == "hold_amber":
        if state.pressure_kpa > 5:
            return None
        return State(state.powered, state.pressure_kpa, False)
    if action == "hold_green":
        new_pressure = state.pressure_kpa + 28
        if not state.powered or not state.clamped or new_pressure > 35:
            return None
        return State(True, new_pressure, True)
    if action == "press_green":
        if not (state.powered and state.clamped and 20 <= state.pressure_kpa <= 30):
            return None
        return State(True, state.pressure_kpa, True, ready=True)
    if action == "finish":
        return None
    raise KeyError(action)


def require_policy(policy: Mapping[State, Mapping[str, float]], state: State) -> Mapping[str, float]:
    try:
        return policy[state]
    except KeyError as error:
        raise KeyError(f"no archived Jev distribution for reachable state {state_label(state)}") from error


@dataclass
class ExactResult:
    horizon: int
    z: float
    first_action_marginals: dict[str, float]
    map_actions: list[str]
    map_probability: float
    valid_trajectories: int
    circuit_states: int
    circuit_edges: int
    inference_ms: float


def circuit_inference(
    policy: Mapping[State, Mapping[str, float]], initial: State, horizon: int
) -> ExactResult:
    """Evaluate the layered trajectory circuit with sum- and max-product."""
    visited_nodes: set[tuple[State, int]] = set()
    visited_edges: set[tuple[State, int, str, State]] = set()

    @lru_cache(maxsize=None)
    def solve(state: State, remaining: int) -> tuple[float, float, tuple[str, ...], int]:
        visited_nodes.add((state, remaining))
        if state.ready:
            return 1.0, 1.0, (), 1
        if remaining == 0:
            return 0.0, 0.0, (), 0
        local = require_policy(policy, state)
        total = 0.0
        best = 0.0
        best_path: tuple[str, ...] = ()
        count = 0
        for action in ACTIONS:
            next_state = transition(state, action)
            if next_state is None:
                continue
            visited_edges.add((state, remaining, action, next_state))
            child_z, child_best, child_path, child_count = solve(next_state, remaining - 1)
            edge_mass = local[action] * child_z
            total += edge_mass
            count += child_count
            candidate = local[action] * child_best
            if candidate > best:
                best = candidate
                best_path = (action,) + child_path
        return total, best, best_path, count

    started = time.perf_counter()
    z, map_probability, map_path, valid_count = solve(initial, horizon)
    local = require_policy(policy, initial)
    first_mass: dict[str, float] = {}
    for action in ACTIONS:
        next_state = transition(initial, action)
        if next_state is None:
            first_mass[action] = 0.0
            continue
        child_z = solve(next_state, horizon - 1)[0]
        first_mass[action] = local[action] * child_z
    elapsed = 1000.0 * (time.perf_counter() - started)
    marginals = {action: mass / z if z else 0.0 for action, mass in first_mass.items()}
    return ExactResult(
        horizon=horizon,
        z=z,
        first_action_marginals=marginals,
        map_actions=list(map_path),
        map_probability=map_probability,
        valid_trajectories=valid_count,
        circuit_states=len(visited_nodes),
        circuit_edges=len(visited_edges),
        inference_ms=elapsed,
    )


def enumerate_trajectories(
    policy: Mapping[State, Mapping[str, float]], initial: State, horizon: int
) -> ExactResult:
    """Independent exhaustive oracle over all legal action prefixes."""
    valid: list[tuple[tuple[str, ...], float]] = []
    first_mass = {action: 0.0 for action in ACTIONS}

    def visit(state: State, remaining: int, path: tuple[str, ...], probability: float) -> None:
        if state.ready:
            valid.append((path, probability))
            first_mass[path[0]] += probability
            return
        if remaining == 0:
            return
        local = require_policy(policy, state)
        for action in ACTIONS:
            next_state = transition(state, action)
            if next_state is not None:
                visit(next_state, remaining - 1, path + (action,), probability * local[action])

    started = time.perf_counter()
    visit(initial, horizon, (), 1.0)
    elapsed = 1000.0 * (time.perf_counter() - started)
    z = sum(probability for _, probability in valid)
    best_path, best_probability = max(valid, key=lambda row: row[1]) if valid else ((), 0.0)
    return ExactResult(
        horizon=horizon,
        z=z,
        first_action_marginals={action: mass / z if z else 0.0 for action, mass in first_mass.items()},
        map_actions=list(best_path),
        map_probability=best_probability,
        valid_trajectories=len(valid),
        circuit_states=0,
        circuit_edges=0,
        inference_ms=elapsed,
    )


def assert_matches(left: ExactResult, right: ExactResult, tolerance: float = 1e-12) -> None:
    if not math.isclose(left.z, right.z, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"Z mismatch: circuit={left.z}, enumeration={right.z}")
    for action in ACTIONS:
        if not math.isclose(
            left.first_action_marginals[action], right.first_action_marginals[action],
            rel_tol=0.0, abs_tol=tolerance,
        ):
            raise AssertionError(
                f"marginal mismatch for {action}: "
                f"circuit={left.first_action_marginals[action]}, "
                f"enumeration={right.first_action_marginals[action]}"
            )
    if not math.isclose(left.map_probability, right.map_probability, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"MAP probability mismatch: circuit={left.map_probability}, enumeration={right.map_probability}"
        )
    if left.map_actions != right.map_actions:
        raise AssertionError(f"MAP path mismatch: circuit={left.map_actions}, enumeration={right.map_actions}")
    if left.valid_trajectories != right.valid_trajectories:
        raise AssertionError(
            f"valid trajectory mismatch: circuit={left.valid_trajectories}, "
            f"enumeration={right.valid_trajectories}"
        )


def raw_greedy_rollout(
    policy: Mapping[State, Mapping[str, float]], initial: State, steps: int
) -> dict:
    state = initial
    trace = []
    for _ in range(steps):
        if state.ready:
            break
        local = require_policy(policy, state)
        action = max(ACTIONS, key=lambda candidate: local[candidate])
        next_state = transition(state, action)
        trace.append({
            "state": state_label(state),
            "action": action,
            "probability": local[action],
            "legal": next_state is not None,
        })
        if next_state is None:
            return {"success": False, "reason": "temporal-rule violation", "trace": trace}
        state = next_state
    return {
        "success": state.ready,
        "reason": "accepted" if state.ready else "step limit / loop",
        "trace": trace,
    }


def myopic_constrained_rollout(
    policy: Mapping[State, Mapping[str, float]], initial: State, steps: int
) -> dict:
    state = initial
    trace = []
    for _ in range(steps):
        if state.ready:
            break
        local = require_policy(policy, state)
        legal = [action for action in ACTIONS if transition(state, action) is not None]
        action = max(legal, key=lambda candidate: local[candidate])
        trace.append({
            "state": state_label(state),
            "action": action,
            "probability": local[action],
            "legal": True,
        })
        state = transition(state, action)  # type: ignore[assignment]
    return {
        "success": state.ready,
        "reason": "accepted" if state.ready else "step limit / loop",
        "trace": trace,
    }


def receding_horizon_rollout(
    policy: Mapping[State, Mapping[str, float]], initial: State, horizon: int, steps: int
) -> dict:
    state = initial
    trace = []
    for _ in range(steps):
        if state.ready:
            break
        inference = circuit_inference(policy, state, horizon)
        if inference.z <= 0:
            return {"success": False, "reason": "no feasible continuation", "trace": trace}
        action = max(ACTIONS, key=lambda candidate: inference.first_action_marginals[candidate])
        local = require_policy(policy, state)
        trace.append({
            "state": state_label(state),
            "action": action,
            "local_probability": local[action],
            "conditioned_marginal": inference.first_action_marginals[action],
            "valid_mass_z": inference.z,
        })
        next_state = transition(state, action)
        if next_state is None:
            raise AssertionError("conditioned policy selected an invalid transition")
        state = next_state
    return {
        "success": state.ready,
        "reason": "accepted" if state.ready else "step limit",
        "trace": trace,
    }


def action_sequence(result: Mapping[str, object]) -> str:
    return " -> ".join(row["action"] for row in result["trace"])  # type: ignore[index]


def render_report(result: Mapping[str, object]) -> str:
    exact = result["exact_inference"]
    rows = []
    for name, label in (
        ("raw_greedy", "Raw Jev greedy"),
        ("myopic_constraints", "One-step constrained argmax"),
        ("multi_step_pcl", "Multi-step PCL"),
    ):
        run = result["rollouts"][name]  # type: ignore[index]
        rows.append(
            f"| {label} | {'yes' if run['success'] else 'no'} | {run['reason']} | "
            f"`{action_sequence(run)}` |"
        )
    marginals = exact["first_action_marginals"]
    marginal_rows = "\n".join(
        f"| `{action}` | {result['initial_local_distribution'][action]:.6f} | {marginals[action]:.6f} |"
        for action in ACTIONS
    )
    decisive = result["decisive_state"]
    decisive_rows = "\n".join(
        f"| `{action}` | {decisive['local_distribution'][action]:.6f} | "
        f"{decisive['first_action_marginals'][action]:.6f} |"
        for action in ACTIONS
    )
    horizon_rows = "\n".join(
        f"| {row['horizon']} | {row['z']:.12f} | {row['valid_trajectories']:,} |"
        for row in result["horizon_sweep"]  # type: ignore[index]
    )
    return f"""# JevAny control-panel multi-step Choice pilot

**Status: PASS.** The layered sum-product/max-product circuit agrees with
complete trajectory enumeration for valid mass, every first-action marginal,
trajectory MAP, and the number of valid trajectories.

## Setup

- Source: JevAny `case-showcase.json`, case `control_panel`
- Archived attempts: {result['source']['attempts']} ({result['source']['successful_attempts']} successful)
- Archived decisions: {result['source']['decisions']}
- Fault-free distributions used to estimate the symbolic Markov policy: {result['source']['fault_free_policy_observations']}
- Initial state: `{result['initial_state']}`
- Planning horizon: {result['horizon']} actions
- Temporal target: reach the accepted state without violating an action precondition

## Exact trajectory inference

- Valid mass $Z$: {exact['z']:.12f}
- Valid trajectories: {exact['valid_trajectories']:,}
- Circuit states / edges: {exact['circuit_states']} / {exact['circuit_edges']}
- Circuit inference: {exact['inference_ms']:.4f} ms
- Exhaustive enumeration: {result['enumeration']['inference_ms']:.4f} ms
- Maximum numerical discrepancy: {result['validation']['max_abs_error']:.3e}
- Joint MAP: `{' -> '.join(exact['map_actions'])}`
- Joint MAP base probability: {exact['map_probability']:.12g}

## First-action probabilities

| Action | Local Jev $p_0$ | Multi-step marginal $q$ |
|---|---:|---:|
{marginal_rows}

## Closed-loop comparison

| Method | Success | Outcome | Actions |
|---|:---:|---|---|
{chr(10).join(rows)}

## The decisive state

After the first action isolates power, the fixture remains at 48 kPa. Local
greedy prefers restoring power, which returns to the initial state. The
multi-step marginal instead prefers bleeding, even though its local
probability is lower.

State: `{decisive['state']}`

| Action | Local Jev $p_0$ | Multi-step marginal $q$ |
|---|---:|---:|
{decisive_rows}

## Horizon sensitivity

| Horizon | Valid mass $Z$ | Valid trajectories |
|---:|---:|---:|
{horizon_rows}

The one-step constrained policy removes immediately unsafe actions, while it
still assigns no value to future reachability. At the isolated 48 kPa state,
its highest-probability legal action restores power and returns to the initial
state. Multi-step conditioning sums the probability of all temporally valid
continuations and selects the lower-local-probability bleed action, which opens
the path to acceptance.

## Temporal rules

1. Bleeding requires isolated power.
2. Closing the clamp requires isolated power and pressure at most 5 kPa.
3. Pumping requires power and a closed clamp; pressure may not exceed 35 kPa.
4. Acceptance requires power, a closed clamp, and pressure in 20--30 kPa.
5. Opening a loaded clamp and finishing before acceptance are infeasible.

The pilot estimates one probability distribution per symbolic state by
averaging archived Jev distributions from that state. This is an offline policy
reconstruction, not a fresh model run or a full release of the original 3D
environment.
"""


def run(source: Path, horizon: int, rollout_steps: int) -> dict:
    policy, source_metadata = load_empirical_policy(source)
    exact = circuit_inference(policy, INITIAL_STATE, horizon)
    oracle = enumerate_trajectories(policy, INITIAL_STATE, horizon)
    assert_matches(exact, oracle)
    decisive_exact = circuit_inference(policy, DECISIVE_STATE, horizon)
    decisive_oracle = enumerate_trajectories(policy, DECISIVE_STATE, horizon)
    assert_matches(decisive_exact, decisive_oracle)
    errors = [abs(exact.z - oracle.z), abs(exact.map_probability - oracle.map_probability)]
    errors.extend(
        abs(exact.first_action_marginals[action] - oracle.first_action_marginals[action])
        for action in ACTIONS
    )
    errors.extend([
        abs(decisive_exact.z - decisive_oracle.z),
        abs(decisive_exact.map_probability - decisive_oracle.map_probability),
    ])
    errors.extend(
        abs(
            decisive_exact.first_action_marginals[action]
            - decisive_oracle.first_action_marginals[action]
        )
        for action in ACTIONS
    )
    result = {
        "source": source_metadata,
        "horizon": horizon,
        "initial_state": state_label(INITIAL_STATE),
        "initial_local_distribution": dict(policy[INITIAL_STATE]),
        "exact_inference": asdict(exact),
        "enumeration": asdict(oracle),
        "decisive_state": {
            "state": state_label(DECISIVE_STATE),
            "local_distribution": dict(policy[DECISIVE_STATE]),
            **asdict(decisive_exact),
        },
        "validation": {"status": "PASS", "max_abs_error": max(errors)},
        "horizon_sweep": [
            {
                "horizon": current_horizon,
                "z": sweep.z,
                "valid_trajectories": sweep.valid_trajectories,
            }
            for current_horizon in range(1, horizon + 3)
            for sweep in [circuit_inference(policy, INITIAL_STATE, current_horizon)]
        ],
        "rollouts": {
            "raw_greedy": raw_greedy_rollout(policy, INITIAL_STATE, rollout_steps),
            "myopic_constraints": myopic_constrained_rollout(policy, INITIAL_STATE, rollout_steps),
            "multi_step_pcl": receding_horizon_rollout(
                policy, INITIAL_STATE, horizon=horizon, steps=rollout_steps
            ),
        },
    }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "tmp/jevany-review/docs/demos/case-showcase.json",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "pilot/results/jevany_control_panel_multistep.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=root / "pilot/results/jevany_control_panel_multistep.md",
    )
    args = parser.parse_args()
    result = run(args.source, args.horizon, args.rollout_steps)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_report.write_text(render_report(result))
    print(json.dumps({
        "validation": result["validation"],
        "z": result["exact_inference"]["z"],
        "valid_trajectories": result["exact_inference"]["valid_trajectories"],
        "joint_map": result["exact_inference"]["map_actions"],
        "rollout_success": {
            name: row["success"] for name, row in result["rollouts"].items()
        },
        "output_json": str(args.output_json),
        "output_report": str(args.output_report),
    }, indent=2))


if __name__ == "__main__":
    main()
