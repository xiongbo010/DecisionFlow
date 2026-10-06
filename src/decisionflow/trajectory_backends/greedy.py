"""Step-wise local greedy baseline for finite trajectory models."""

from __future__ import annotations

import time

from ..trajectory import TrajectoryResult, TrajectorySpec
from .common import normalized_outcomes, normalized_policy


class GreedyTrajectoryBackend:
    """Take the locally most likely action and outcome at every visited state."""

    name = "greedy"
    exact = False
    capabilities = ("point_prediction", "terminal_check")

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        started = time.perf_counter()
        terminals = set(spec.terminal_states)
        state = spec.initial_state
        path = []
        visited = set()
        edges = 0
        failed_action = None
        for remaining in range(spec.horizon, 0, -1):
            visited.add((state, remaining))
            if state in terminals:
                break
            local = normalized_policy(spec, state)
            action = max(spec.actions, key=lambda item: local[item])
            outcomes = normalized_outcomes(
                spec.transitions.get(state, {}).get(action, ())
            )
            if local[action] <= 0 or not outcomes:
                failed_action = action
                break
            edges += len(outcomes)
            outcome = max(outcomes, key=lambda item: item.probability)
            state = outcome.next_state
            path.append((action, state))
        terminal_reached = state in terminals
        elapsed = 1000.0 * (time.perf_counter() - started)
        return TrajectoryResult(
            valid_mass=None,
            first_action_marginals={},
            trajectory_map=tuple(path),
            map_probability=None,
            states_visited=len(visited),
            edges_evaluated=edges,
            inference_ms=elapsed,
            exact=False,
            backend=self.name,
            prediction_kind="stepwise_greedy",
            capabilities=self.capabilities,
            diagnostics={
                "terminal_reached": terminal_reached,
                "final_state": state,
                "failed_action": failed_action,
                "note": "The returned path is a local point prediction, not a trajectory MAP.",
            },
        )
