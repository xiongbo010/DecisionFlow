from __future__ import annotations

import functools
import time

from ..errors import UnsatisfiableError
from ..trajectory import TrajectoryResult, TrajectorySpec
from .common import normalized_outcomes, normalized_policy


class DynamicProgrammingTrajectoryBackend:
    name = "dynamic_programming"
    exact = True
    capabilities = ("valid_mass", "first_action_marginals", "joint_map")

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        terminals = set(spec.terminal_states)
        visited = set()
        edges = set()

        @functools.lru_cache(maxsize=None)
        def solve(state: str, remaining: int):
            visited.add((state, remaining))
            if state in terminals:
                return 1.0, 1.0, ()
            if remaining == 0:
                return 0.0, 0.0, ()
            local = normalized_policy(spec, state)
            total = 0.0
            best = 0.0
            best_path = ()
            for action in spec.actions:
                action_probability = local[action]
                if action_probability <= 0:
                    continue
                for outcome in normalized_outcomes(
                    spec.transitions.get(state, {}).get(action, ())
                ):
                    edges.add((state, remaining, action, outcome.next_state))
                    child_z, child_best, child_path = solve(
                        outcome.next_state, remaining - 1
                    )
                    branch = action_probability * outcome.probability
                    total += branch * child_z
                    candidate = branch * child_best
                    if candidate > best:
                        best = candidate
                        best_path = ((action, outcome.next_state),) + child_path
            return total, best, best_path

        started = time.perf_counter()
        z, best, best_path = solve(spec.initial_state, spec.horizon)
        if z <= 0:
            raise UnsatisfiableError("no terminal trajectory has positive probability")
        first_mass = {action: 0.0 for action in spec.actions}
        local = normalized_policy(spec, spec.initial_state)
        if spec.horizon > 0:
            for action in spec.actions:
                for outcome in normalized_outcomes(
                    spec.transitions.get(spec.initial_state, {}).get(action, ())
                ):
                    child_z = solve(outcome.next_state, spec.horizon - 1)[0]
                    first_mass[action] += local[action] * outcome.probability * child_z
        elapsed = 1000.0 * (time.perf_counter() - started)
        return TrajectoryResult(
            valid_mass=z,
            first_action_marginals={
                action: value / z for action, value in first_mass.items()
            },
            trajectory_map=best_path,
            map_probability=best / z,
            states_visited=len(visited),
            edges_evaluated=len(edges),
            inference_ms=elapsed,
            exact=True,
            backend=self.name,
            capabilities=self.capabilities,
        )
