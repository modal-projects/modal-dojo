"""Log custom scalar metrics to the current Training Gym run's dashboard.

Use inside a training worker or callback, where the launcher supplies the run
context and reporting credentials. No ``init()`` or W&B installation is needed::

    from modal_training_gym import metric_collector

    metric_collector.log({"custom/reward": 0.8}, step=10)

These calls go directly to Training Gym, regardless of the configured external
metric provider. They do not forward to W&B or Trackio.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

__all__ = ["log"]


def log(
    data: Mapping[str, Any],
    step: int | None = None,
    commit: bool | None = None,
) -> None:
    """Queue scalar metrics for the current run's Metrics workspace.

    Nested mappings flatten to slash-separated keys. Only finite numeric values
    (including scalar tensors) are captured; strings, media and booleans are
    ignored. Calls share the framework's metric buffer and implicit step counter,
    so pass an explicit ``step`` to align custom metrics with training steps.
    With no explicit step, ``commit=False`` keeps the current implicit step open.

    Delivery is asynchronous and best-effort, with a final flush at worker exit.
    Calls are ignored outside a Training Gym worker or when ``metrics=None``.
    """
    if not os.environ.get("TRAINING_GYM_METRIC_PROVIDER") or not os.environ.get(
        "TRAINING_GYM_TRAINING_RUN_ID"
    ):
        return

    from modal_training_gym.common.metric_mirror import mirror_log

    mirror_log(data, step=step, commit=commit)
