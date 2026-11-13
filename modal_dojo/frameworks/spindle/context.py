"""What the driver hands the program it runs, besides a Tinker client."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modal_dojo.common.run import TrainingRun


class TinkerContext:
    """What a program gets besides its client.

    ``log_dir`` is the run's directory on the ``spindle-checkpoints`` volume.
    It survives retries, so a program that saves progress there can resume.
    """

    def __init__(self, run: TrainingRun, log_dir: str):
        self.run = run
        self.log_dir = log_dir

    def log_metrics(self, step: int, **metrics: float) -> None:
        """Chart finite scalar metrics at ``step`` in the Dojo dashboard."""
        from modal_dojo.common.reporting import _enqueue_metric_points

        scalars = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and math.isfinite(value)
        }
        if not scalars:
            return
        self.run.metrics = {**self.run.metrics, **scalars}
        _enqueue_metric_points(
            {
                "training_run_id": self.run.training_run_id,
                "points": [{"step": step, "metrics": scalars, "time": time.time()}],
            }
        )
