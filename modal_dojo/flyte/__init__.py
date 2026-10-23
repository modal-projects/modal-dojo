"""Flyte 2 backend for modal-dojo.

Keeps the user-facing ``TrainConfig`` API but runs the training pipeline
(download model, prepare dataset, convert checkpoint, train) as Flyte 2 tasks
on a Flyte cluster — the local devbox (``flyte start devbox``) or Union.

The config objects (recipes, model presets, datasets) are backend-agnostic and
are reused verbatim; only orchestration lives here. See ``flyte/DESIGN.md`` and
``flyte/design/*.md`` for the full design.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """Which cluster runtime a ``TrainConfig`` launches on."""

    FLYTE = "flyte"
    MODAL = "modal"


class TrainingMode(str, Enum):
    """How heavy the ``train`` stage is.

    - ``MINIMAL`` runs a small, self-contained HuggingFace ``Trainer`` on CPU
      (or one GPU) so the whole pipeline is verifiable on a devbox with no
      framework image or GPU cluster.
    - ``SLIME`` / ``MILES`` run the real framework launcher on GPU resources.
    """

    MINIMAL = "minimal"
    SLIME = "slime"
    MILES = "miles"


__all__ = [
    "Backend",
    "TrainingMode",
    "FlyteTrainingRun",
    "build_pipeline",
    "flyte_config_path",
    "flyte_launch",
    "flyte_train",
]

from modal_dojo.flyte.launch import (  # noqa: E402
    flyte_config_path,
    flyte_launch,
    flyte_train,
)
from modal_dojo.flyte.pipeline import build_pipeline  # noqa: E402
from modal_dojo.flyte.run import FlyteTrainingRun  # noqa: E402
