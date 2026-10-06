from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple

from ..trajectory import TrajectorySpec, Transition


def normalized_policy(spec: TrajectorySpec, state: str) -> Dict[str, float]:
    source = spec.policy.get(state, {})
    row = {action: max(0.0, float(source.get(action, 0.0))) for action in spec.actions}
    total = sum(row.values())
    if total <= 0:
        return row
    return {action: value / total for action, value in row.items()}


def normalized_outcomes(outcomes: Iterable[Transition]) -> Tuple[Transition, ...]:
    rows = tuple(outcomes)
    total = sum(max(0.0, item.probability) for item in rows)
    if total <= 0:
        return ()
    return tuple(
        Transition(item.next_state, max(0.0, item.probability) / total) for item in rows
    )


def draw_key(row, ordered_keys, rng: random.Random):
    threshold = rng.random()
    cumulative = 0.0
    for key in ordered_keys:
        cumulative += row.get(key, 0.0)
        if threshold <= cumulative:
            return key
    return ordered_keys[-1]
