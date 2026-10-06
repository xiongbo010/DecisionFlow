from __future__ import annotations

import time

from ..errors import UnsatisfiableError
from ..trajectory import TrajectoryResult, TrajectorySpec
from .common import normalized_outcomes, normalized_policy


class BeamSearchTrajectoryBackend:
    name = "beam_search"
    exact = False
    capabilities = ("joint_map",)

    def __init__(self, width: int = 16):
        if width <= 0:
            raise ValueError("beam width must be positive")
        self.width = int(width)

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        started = time.perf_counter()
        terminals = set(spec.terminal_states)
        beam = [(1.0, spec.initial_state, ())]
        completed = []
        expanded = 0
        edges = 0
        for _ in range(spec.horizon):
            candidates = []
            for probability, state, path in beam:
                if state in terminals:
                    completed.append((probability, state, path))
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
                        candidates.append(
                            (
                                probability * local[action] * outcome.probability,
                                outcome.next_state,
                                path + ((action, outcome.next_state),),
                            )
                        )
            candidates.sort(key=lambda item: (-item[0], item[2]))
            beam = candidates[: self.width]
            if not beam:
                break
        completed.extend(item for item in beam if item[1] in terminals)
        if not completed:
            raise UnsatisfiableError(
                "beam search found no terminal trajectory; increase width or use an exact backend"
            )
        probability, _, path = max(completed, key=lambda item: (item[0], item[2]))
        elapsed = 1000.0 * (time.perf_counter() - started)
        return TrajectoryResult(
            valid_mass=None,
            first_action_marginals={},
            trajectory_map=path,
            map_probability=None,
            states_visited=expanded,
            edges_evaluated=edges,
            inference_ms=elapsed,
            exact=False,
            backend=self.name,
            capabilities=self.capabilities,
            diagnostics={
                "width": self.width,
                "map_base_probability": probability,
            },
        )
