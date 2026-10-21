from __future__ import annotations

from enum import Enum
from typing import TypeAlias


class SlimeStatus(str, Enum):
    INITIALIZING = "initializing"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREPARE_DATASET = "prepare_dataset"
    ROLLOUT_INITIALIZING = "initialize_rollouts"
    ROLLOUT_LOGGING = "generate_rollouts"
    EVAL_ROLLOUT_LOGGING = "evaluate_rollouts"
    COMPUTE_LOG_PROBS = "compute_log_probs"
    OPTIMIZER_STEP = "optimizer_step"  # before train step
    WEIGHT_SYNC = "weight_sync"
    OFFLOAD_ROLLOUT = "offload_rollout"
    OFFLOAD_TRAIN = "offload_train"
    CHECKPOINT_SAVE = "checkpoint_save"
    TRAINING = "training"


class MilesStatus(str, Enum):
    INITIALIZING = "initializing"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL = "convert_model"
    PREPARE_DATASET = "prepare_dataset"
    ROLLOUT_INITIALIZING = "initialize_rollouts"
    ROLLOUT_LOGGING = "generate_rollouts"
    EVAL_ROLLOUT_LOGGING = "evaluate_rollouts"
    COMPUTE_LOG_PROBS = "compute_log_probs"
    OPTIMIZER_STEP = "optimizer_step"  # before train step
    WEIGHT_SYNC = "weight_sync"
    OFFLOAD_ROLLOUT = "offload_rollout"
    OFFLOAD_TRAIN = "offload_train"
    CHECKPOINT_SAVE = "checkpoint_save"
    TRAINING = "training"


class SpindleStatus(str, Enum):
    INITIALIZING = "initializing"
    ROLLOUT_LOGGING = "generate_rollouts"
    TRAINING = "training"


FrameworkStatus: TypeAlias = SlimeStatus | MilesStatus | SpindleStatus


def resolve_framework_status(phase: str, framework: str) -> FrameworkStatus | None:
    """Parse a reported phase string into the framework's status enum.

    Returns ``None`` for a phase the framework doesn't know.
    """
    enums = {"miles": MilesStatus, "slime": SlimeStatus, "spindle": SpindleStatus}
    if framework.strip().lower() not in enums:
        raise ValueError(f"Invalid framework string detected: {framework}")

    status_enum = enums[framework.strip().lower()]
    try:
        return status_enum(phase.strip())
    except ValueError:
        return None
