"""Domain-specific errors with a stable infrastructure/candidate distinction."""


class NeuroGolfError(Exception):
    """Base class for expected NeuroGolf errors."""


class ConfigurationError(NeuroGolfError):
    """Configuration is absent, malformed, or inconsistent."""


class MappingError(NeuroGolfError):
    """The canonical task mapping or a mapped source is invalid."""


class GeneratorError(NeuroGolfError):
    """ARC-GEN import or execution failed."""


class IntegrityError(NeuroGolfError):
    """A protected artifact does not match its integrity manifest."""


class CandidateError(NeuroGolfError):
    """A candidate model is absent, illegal, or fails at runtime."""


class GateInfrastructureError(NeuroGolfError):
    """The acceptance judge could not complete reliably."""


class PromotionError(NeuroGolfError):
    """A champion lifecycle operation is not eligible or could not commit."""
