class DojoError(ValueError):
    """Base error for Modal Dojo."""


class DojoConfigError(DojoError):
    """Raised when a training or deploy config is invalid."""


class GpuAllocationError(DojoConfigError):
    """Raised when a recipe's cluster or parallelism settings are invalid."""
