from __future__ import annotations

import heapq
import math
import time

from ..errors import UnsatisfiableError
from ..trajectory import TrajectoryResult, TrajectorySpec
from .common import normalized_outcomes, normalized_policy


class BestFirstTrajectoryBackend:
    """Exact MAP search with state/depth dominance on a finite horizon."""

    name = "best_first"
    exact = True
    capabilities = ("joint_map",)

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        started = time.perf_counter()
        terminals = set(spec.terminal_states)
        queue = [(0.0, (), spec.initial_state, spec.horizon)]
        best_cost = {(spec.initial_state, spec.horizon): 0.0}
        expanded = 0
        edges = 0
        while queue:
            cost, path, state, remaining = heapq.heappop(queue)
            key = (state, remaining)
            if cost > best_cost.get(key, math.inf) + 1e-15:
                continue
            if state in terminals:
                elapsed = 1000.0 * (time.perf_counter() - started)
                return TrajectoryResult(
                    valid_mass=None,
                    first_action_marginals={},
                    trajectory_map=path,
                    map_probability=None,
                    states_visited=expanded,
                    edges_evaluated=edges,
                    inference_ms=elapsed,
                    exact=True,
                    backend=self.name,
                    capabilities=self.capabilities,
                    diagnostics={"map_base_probability": math.exp(-cost)},
                )
            if remaining == 0:
                continue
            expanded += 1
            local = normalized_policy(spec, state)
            for action in spec.actions:
                if local[action] <= 0:
                    continue
                for outcome in normalized_outcomes(
                    spec.transitions.get(state, {}).get(action, ())
                ):
                    edges += 1
                    branch = local[action] * outcome.probability
                    candidate = cost - math.log(branch)
                    child_key = (outcome.next_state, remaining - 1)
                    if candidate + 1e-15 < best_cost.get(child_key, math.inf):
                        best_cost[child_key] = candidate
                        heapq.heappush(
                            queue,
                            (
                                candidate,
                                path + ((action, outcome.next_state),),
                                outcome.next_state,
                                remaining - 1,
                            ),
                        )
        raise UnsatisfiableError("no terminal trajectory has positive probability")
