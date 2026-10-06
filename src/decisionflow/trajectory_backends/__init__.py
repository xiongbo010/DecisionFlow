"""Built-in inference methods for finite trajectory models."""

from __future__ import annotations

from ..backends.registry import BackendRegistry
from .beam import BeamSearchTrajectoryBackend
from .best_first import BestFirstTrajectoryBackend
from .dynamic_programming import DynamicProgrammingTrajectoryBackend
from .greedy import GreedyTrajectoryBackend
from .sampling import MonteCarloTrajectoryBackend


def create_trajectory_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    full_queries = ("valid_mass", "first_action_marginals", "joint_map")
    registry.register(
        "greedy",
        lambda **options: GreedyTrajectoryBackend(**options),
        capabilities=GreedyTrajectoryBackend.capabilities,
        exact=False,
        description="Step-wise local action and transition argmax.",
        aliases=("local_argmax",),
    )
    registry.register(
        "dynamic_programming",
        lambda **options: DynamicProgrammingTrajectoryBackend(**options),
        capabilities=full_queries,
        exact=True,
        description="Exact finite-horizon sum-product and max-product recursion.",
        aliases=("dp", "exact", "auto"),
    )
    registry.register(
        "best_first",
        lambda **options: BestFirstTrajectoryBackend(**options),
        capabilities=("joint_map",),
        exact=True,
        description="Exact maximum-probability terminal trajectory search.",
        aliases=("astar", "uniform_cost"),
    )
    registry.register(
        "beam_search",
        lambda **options: BeamSearchTrajectoryBackend(**options),
        capabilities=("joint_map",),
        exact=False,
        description="Finite-width approximate trajectory MAP search.",
        aliases=("beam",),
    )
    registry.register(
        "monte_carlo",
        lambda **options: MonteCarloTrajectoryBackend(**options),
        capabilities=full_queries,
        exact=False,
        description="Rollout estimates of terminal mass, marginals, and MAP.",
        aliases=("sampling", "trajectory_sampling"),
    )
    return registry


__all__ = [
    "BeamSearchTrajectoryBackend",
    "BestFirstTrajectoryBackend",
    "DynamicProgrammingTrajectoryBackend",
    "GreedyTrajectoryBackend",
    "MonteCarloTrajectoryBackend",
    "create_trajectory_backend_registry",
]
