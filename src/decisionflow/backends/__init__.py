from .auto import AutoBackend
from .base import InferenceBackend
from .enumeration import EnumerationBackend
from .greedy import GreedyBackend
from .registry import BackendDescriptor, BackendRegistry
from .sampling import RejectionSamplingBackend


def _sdd_factory(**options):
    from .sdd import SDDBackend

    return SDDBackend(**options)


def create_backend_registry() -> BackendRegistry:
    """Create an isolated registry populated with DecisionFlow built-ins."""
    registry = BackendRegistry()
    full_queries = ("valid_mass", "marginals", "joint_map")
    registry.register(
        "greedy",
        lambda **options: GreedyBackend(**options),
        capabilities=GreedyBackend.capabilities,
        exact=False,
        description="Independent local argmax with a post-hoc feasibility check.",
        aliases=("local_argmax",),
    )
    registry.register(
        "auto",
        lambda **options: AutoBackend(**options),
        capabilities=full_queries,
        exact=True,
        description="Enumeration for small worlds and SDD compilation for larger ones.",
    )
    registry.register(
        "enumeration",
        lambda **options: EnumerationBackend(**options),
        capabilities=full_queries,
        exact=True,
        description="Transparent exhaustive reference inference.",
        aliases=("exact_enumeration",),
    )
    registry.register(
        "sdd",
        _sdd_factory,
        capabilities=full_queries,
        exact=True,
        description="Reusable SDD probabilistic-circuit compilation.",
        aliases=("pc", "probabilistic_circuit"),
    )
    registry.register(
        "rejection_sampling",
        lambda **options: RejectionSamplingBackend(**options),
        capabilities=("valid_mass", "marginals", "joint_map"),
        exact=False,
        description="Approximate rejection and importance sampling.",
        aliases=("sampling", "monte_carlo"),
    )
    return registry


__all__ = [
    "AutoBackend",
    "BackendDescriptor",
    "BackendRegistry",
    "EnumerationBackend",
    "GreedyBackend",
    "InferenceBackend",
    "RejectionSamplingBackend",
    "create_backend_registry",
]
