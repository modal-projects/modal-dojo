"""The Spindle run record, each attempt's outcome, and its dashboard reports."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from enum import Enum
import logging
from pathlib import Path
import time

from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.framework import Framework
from modal_dojo.common.launcher_helpers import mark_run_failed, mark_run_stopped
from modal_dojo.common.launcher_utils import (
    is_sensitive_recipe_field,
    serialize_recipe_param_value,
    serialize_recipe_value,
)
from modal_dojo.common.run import (
    TrainingRun,
    TrainingRunStatus,
    mark_training_attempt_finished,
    mark_training_attempt_started,
)
from modal_dojo.common.status import SpindleStatus

logger = logging.getLogger(__name__)


def _recipe_fields(config, prefix=""):
    """Every config field, with nested configs and datasets as dotted keys."""
    fields = {}
    names = getattr(config, "__chz_fields__", None) or vars(config)
    for name in names:
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        key = prefix + name
        if hasattr(value, "__chz_fields__") or isinstance(value, DatasetConfig):
            fields[key] = serialize_recipe_value(type(value))
            fields.update(_recipe_fields(value, f"{key}."))
        else:
            fields[key] = _field_value(name, value)
    return fields


def _field_value(name, value):
    """Plain values as is; objects such as evaluators or tasks as their type.

    Their repr can hold every task's text and would skip redaction.
    """
    if is_sensitive_recipe_field(name) or isinstance(
        value, str | int | float | bool | Enum | Path | dict | type(None)
    ):
        return serialize_recipe_param_value(name, value)
    if isinstance(value, list | tuple | set):
        return [_field_value(name, item) for item in value]
    if callable(value) and hasattr(value, "__qualname__"):
        return serialize_recipe_value(value)
    return serialize_recipe_value(type(value))


def new_training_run(program, training_run_id, log_dir):
    """The run record ``launch()`` saves before the driver container starts."""
    import chz

    from .cookbook import CookbookRL

    if isinstance(program, CookbookRL):
        config = chz.replace(program.config, log_path=log_dir)
        summary = {
            "recipe": _recipe_fields(config),
            "dataset": {"type": type(config.dataset_builder).__name__},
        }
    else:
        summary = {
            "recipe": {
                "program": serialize_recipe_value(type(program)),
                **_recipe_fields(program),
            }
        }
    now = int(time.time())
    return TrainingRun(
        training_run_id=training_run_id,
        framework=Framework.SPINDLE,
        framework_status=SpindleStatus.INITIALIZING,
        config={"model": {"model_name": program.model_name}, **summary},
        source_model={"model_name": program.model_name},
        created_at=now,
        started_at=now,
        metadata={"log_path": log_dir},
    )


@asynccontextmanager
async def run_lifecycle(run: TrainingRun):
    """Record this attempt on the run ``launch()`` saved, then its outcome."""
    # A retried container resumes the same run.
    mark_training_attempt_started(run, started_at=int(time.time()))
    await run.save(is_async=True)
    try:
        yield
    except BaseException as exc:
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
            # Stopping the Modal app cancels the driver.
            mark_run_stopped(run)
        else:
            mark_run_failed(run, exc)
        _end(run)
        await _drain_reports()
        try:
            await run.save(is_async=True)
        except Exception:
            logger.exception("Could not persist terminal Spindle run status")
        raise
    run.status = TrainingRunStatus.COMPLETED
    run.completed_at = _end(run)
    await _drain_reports()
    mark_training_attempt_finished(run, status="completed", ended_at=run.completed_at)
    await run.save(is_async=True)


async def _drain_reports() -> None:
    """Land queued dashboard writes, which rewrite the run, before its final save."""
    from modal_dojo.common.reporting import drain

    try:
        await asyncio.to_thread(drain)
    except Exception:
        logger.exception("Could not drain Dojo reports")


def _end(run: TrainingRun) -> int:
    run.ended_at = ended_at = int(time.time())
    run.duration_seconds = ended_at - run.started_at
    return ended_at


def report_status(run: TrainingRun, status: SpindleStatus, **fields) -> None:
    """Set the run's phase, locally and on the dashboard."""
    from modal_dojo.common.reporting import _enqueue
    from modal_dojo.common.run import FrameworkStatusUpdate

    payload = {
        "training_run_id": run.training_run_id,
        "phase": status.value,
        **fields,
    }
    run.apply_framework_status(FrameworkStatusUpdate(**payload))
    _enqueue(payload)


# Reporting must not stop training, so these log failures instead of raising.
def report_rollouts(run: TrainingRun, step: int, samples: list, metrics: dict) -> None:
    """Send ``step``'s rollout samples, replacing any already sent for it."""
    from modal_dojo.common.reporting import _enqueue_rollout
    from modal_dojo.common.training_rollout import TrainingRolloutResult

    if not samples:
        return
    try:
        result = TrainingRolloutResult(
            training_run_id=run.training_run_id,
            rollout_id=step,
            created_at=int(time.time()),
            samples=samples,
            metrics=metrics,
        )
        payload = result.model_dump(mode="json")
        run.record_latest_rollout(result)
        _enqueue_rollout(payload)
    except Exception:
        logger.exception("Could not report rollouts for step %s", step)


def report_timing(run: TrainingRun, step: int, spans: list[dict]) -> None:
    """Send ``step``'s timing from tinker-cookbook-style span dicts."""
    from modal_dojo.common.reporting import _enqueue_timing

    from .timing import timing_record

    if not spans:
        return
    try:
        record = timing_record(
            run.training_run_id,
            step,
            spans,
            min(span["wall_start"] for span in spans),
        )
        # storage_key is a property, so model_dump leaves it out; the
        # transport keys exit retries by it.
        _enqueue_timing(
            {**record.model_dump(mode="json"), "storage_key": record.storage_key},
            final=True,
        )
    except Exception:
        logger.exception("Could not report timing for step %s", step)
