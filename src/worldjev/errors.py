class WorldJevError(Exception):
    """Base class for WorldJev errors."""


class ConstraintSyntaxError(WorldJevError):
    """A declarative constraint cannot be parsed or typed."""


class InvalidProbabilityError(WorldJevError):
    """A scorer returned a missing, negative, or zero-mass distribution."""


class UnsatisfiableError(WorldJevError):
    """No positive-mass assignment satisfies the grounded model."""


class BackendUnavailableError(WorldJevError):
    """An optional inference backend is unavailable."""


class StatePathError(WorldJevError):
    """A constraint references a state path that cannot be resolved."""

