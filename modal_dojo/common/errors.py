class TrainingGymError(ValueError):
    """Base error for Training Gym."""


class TrainingGymConfigError(TrainingGymError):
    """Raised when a training or deploy config is invalid."""


class GpuAllocationError(TrainingGymConfigError):
    """Raised when a recipe's cluster or parallelism settings are invalid."""
