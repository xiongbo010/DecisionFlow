from __future__ import annotations

from typing import Protocol, Tuple

from ..trajectory import TrajectoryResult, TrajectorySpec


class TrajectoryBackend(Protocol):
    name: str
    exact: bool
    capabilities: Tuple[str, ...]

    def infer(self, spec: TrajectorySpec) -> TrajectoryResult: ...
