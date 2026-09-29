class DojoError(ValueError):
    """Base error for Training Gym."""


class DojoConfigError(DojoError):
    """Raised when a training or deploy config is invalid."""


class GpuAllocationError(DojoConfigError):
    """Raised when a recipe's cluster or parallelism settings are invalid."""
