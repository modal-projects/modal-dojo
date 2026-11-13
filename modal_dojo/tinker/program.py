"""Your own Tinker training loop, run on Modal against Spindle by SpindleRecipe."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from modal_dojo.common.status import SpindleStatus

if TYPE_CHECKING:
    import tinker

    from modal_dojo.common.run import TrainingRun


# Each phase's run status, and the tinker-cookbook span name its timing maps from.
_PHASES = {
    "rollout": (
        SpindleStatus.DO_GROUP_ROLLOUT_AND_FILTER_CONSTANT_REWARD,
        "sampling",
    ),
    "train": (SpindleStatus.TRAIN_STEP, "train_step"),
    "checkpoint": (
        SpindleStatus.SAVE_CHECKPOINT_AND_GET_SAMPLING_CLIENT,
        "save_checkpoint_and_get_sampling_client",
    ),
    # Evaluation has no run status of its own; it only shows in the timing.
    "eval": (None, "run_evaluations_parallel"),
}


class TinkerContext:
    """What a program gets besides its client: Dojo reporting and a log directory.

    ``log_metrics``, ``phase`` and ``log_rollouts`` feed the run's metrics,
    status, per-step timing and rollouts in the Dojo dashboard. ``log_dir`` is
    the run's directory on the ``spindle-checkpoints`` volume. It survives
    retries, so a program that saves progress there can resume.
    """

    def __init__(self, run: TrainingRun, log_dir: str):
        self.run = run
        self.log_dir = log_dir
        self._spans: dict[int, list[dict[str, Any]]] = {}

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

    @contextmanager
    def phase(self, name: str, step: int) -> Iterator[None]:
        """Mark a block as ``step``'s ``"rollout"``, ``"train"``, ``"checkpoint"`` or ``"eval"``.

        The run's status shows the phase while the block runs (evaluation
        excepted), and the block's time appears in the step's timing view.
        """
        if name not in _PHASES:
            raise ValueError(f"phase must be one of {sorted(_PHASES)}, not {name!r}")
        from modal_dojo.frameworks.spindle.reporting import report_status, report_timing

        status, span_name = _PHASES[name]
        if status is not None and self.run.framework_status != status:
            report_status(self.run, status, is_active=True, rollout_id=step)
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            spans = self._spans.setdefault(step, [])
            spans.append(
                {
                    "name": span_name,
                    "wall_start": start,
                    "wall_end": end,
                    "duration": end - start,
                }
            )
            report_timing(self.run, step, spans)

    def log_rollouts(
        self,
        step: int,
        samples: Iterable[Mapping[str, Any]],
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Show ``step``'s training rollouts in the Dojo dashboard.

        Each sample has ``prompt`` and ``response`` text and a numeric
        ``score``, and optionally ``group`` (an index shared by samples drawn
        for the same prompt) and ``metadata``. Logging a step again replaces it.
        """
        from modal_dojo.common.training_rollout import TrainingRolloutSample
        from modal_dojo.frameworks.spindle.reporting import report_rollouts

        built = []
        for index, sample in enumerate(samples):
            missing = {"prompt", "response", "score"} - set(sample)
            if missing:
                raise ValueError(f"rollout sample {index} is missing {sorted(missing)}")
            built.append(
                TrainingRolloutSample(
                    rollout_index=index,
                    sample_index=index,
                    group_index=sample.get("group"),
                    prompt=str(sample["prompt"]),
                    response=str(sample["response"]),
                    score=float(sample["score"]),
                    metadata=dict(sample.get("metadata") or {}),
                )
            )
        if built:
            report_rollouts(self.run, step, built, dict(metrics or {}))


@runtime_checkable
class TinkerProgram(Protocol):
    """A training loop that ``SpindleRecipe`` runs on Modal against Spindle."""

    # Picks the Spindle deployment and is checked against the server.
    model_name: str

    async def run(self, client: tinker.ServiceClient, ctx: TinkerContext) -> Any:
        """Train with ``client``, which already points at Spindle."""
        ...
