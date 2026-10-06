from __future__ import annotations

import random
import time
from collections import Counter
from typing import Optional

from ..errors import UnsatisfiableError
from ..trajectory import TrajectoryResult, TrajectorySpec
from .common import draw_key, normalized_outcomes, normalized_policy


class MonteCarloTrajectoryBackend:
    name = "monte_carlo"
    exact = False
    capabilities = ("valid_mass", "first_action_marginals", "joint_map")

    def __init__(self, samples: int = 10_000, seed: Optional[int] = 0):
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.samples = int(samples)
        self.seed = seed

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult:
        started = time.perf_counter()
        rng = random.Random(self.seed)
        terminals = set(spec.terminal_states)
        accepted = 0
        first = Counter()
        paths = Counter()
        states = set()
        edges = set()
        for _ in range(self.samples):
            state = spec.initial_state
            path = []
            for remaining in range(spec.horizon, 0, -1):
                states.add((state, remaining))
                if state in terminals:
                    break
                local = normalized_policy(spec, state)
                if sum(local.values()) <= 0:
                    break
                action = draw_key(local, spec.actions, rng)
                outcomes = normalized_outcomes(
                    spec.transitions.get(state, {}).get(action, ())
                )
                if not outcomes:
                    break
                outcome_row = {
                    index: item.probability for index, item in enumerate(outcomes)
                }
                index = draw_key(outcome_row, tuple(range(len(outcomes))), rng)
                outcome = outcomes[index]
                edges.add((state, remaining, action, outcome.next_state))
                state = outcome.next_state
                path.append((action, state))
            if state in terminals:
                accepted += 1
                frozen = tuple(path)
                paths[frozen] += 1
                if frozen:
                    first[frozen[0][0]] += 1
        if accepted == 0:
            raise UnsatisfiableError(
                "trajectory sampling found no terminal rollout; increase samples"
            )
        best_path, best_count = max(paths.items(), key=lambda item: (item[1], item[0]))
        elapsed = 1000.0 * (time.perf_counter() - started)
        return TrajectoryResult(
            valid_mass=accepted / self.samples,
            first_action_marginals={
                action: first[action] / accepted for action in spec.actions
            },
            trajectory_map=best_path,
            map_probability=best_count / accepted,
            states_visited=len(states),
            edges_evaluated=len(edges),
            inference_ms=elapsed,
            exact=False,
            backend=self.name,
            capabilities=self.capabilities,
            diagnostics={
                "samples": self.samples,
                "accepted_samples": accepted,
                "acceptance_rate": accepted / self.samples,
            },
        )
