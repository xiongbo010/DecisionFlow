class DecisionFlowError(Exception):
    """Base class for DecisionFlow errors."""


class ConstraintSyntaxError(DecisionFlowError):
    """A declarative constraint cannot be parsed or typed."""


class InvalidProbabilityError(DecisionFlowError):
    """A scorer returned a missing, negative, or zero-mass distribution."""


class UnsatisfiableError(DecisionFlowError):
    """No positive-mass assignment satisfies the grounded model."""


class BackendUnavailableError(DecisionFlowError):
    """An optional inference backend is unavailable."""


class StatePathError(DecisionFlowError):
    """A constraint references a state path that cannot be resolved."""
