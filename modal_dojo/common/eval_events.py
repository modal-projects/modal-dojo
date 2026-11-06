"""POC (option E): push ``checkpoint_saved`` events instead of polling.

Not for merge. The trainer's existing ``checkpoint_save`` phase report also
puts a small event on a named ``modal.Queue``, partitioned by training run.
A subscriber (here PR 777's watcher; in the end, Modal Evals) blocks on the
queue instead of sleeping between Volume listings.

The framework hook fires *before* the checkpoint is committed, so an event is
a wake-up, not proof the weights are readable: the subscriber still confirms
the checkpoint on the Volume before it evaluates.

Event contract::

    {"type": "checkpoint_saved", "version": 1, "training_run_id": "...",
     "app_name": "...", "rollout_id": 39, "event_ts": 1793377481.0}
"""

from __future__ import annotations

import time
from typing import Any

QUEUE_NAME = "modal-evals-checkpoint-events"
CHECKPOINT_SAVE_PHASE = "checkpoint_save"


def _queue() -> Any:
    import modal

    return modal.Queue.from_name(QUEUE_NAME, create_if_missing=True)


def checkpoint_saved_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The event for a phase-report payload, or ``None`` if it is not a
    checkpoint save. Pure."""
    if payload.get("phase") != CHECKPOINT_SAVE_PHASE:
        return None
    training_run_id = payload.get("training_run_id")
    if not training_run_id:
        return None
    return {
        "type": "checkpoint_saved",
        "version": 1,
        "training_run_id": training_run_id,
        "app_name": payload.get("app_name", ""),
        "rollout_id": payload.get("rollout_id"),
        "event_ts": payload.get("event_ts") or time.time(),
    }


def maybe_emit_checkpoint_saved(payload: dict[str, Any]) -> None:
    """Called from the trainer's phase reporter. Never raises or blocks
    training for more than a second."""
    event = checkpoint_saved_event(payload)
    if event is None:
        return
    try:
        _queue().put(event, partition=event["training_run_id"], timeout=1.0)
    except Exception as exc:
        print(f"WARNING: checkpoint_saved event not sent: {exc}")


def wait_for_checkpoint_event(
    training_run_id: str, timeout: float
) -> dict[str, Any] | None:
    """Block until this run saves a checkpoint, or ``timeout`` seconds pass."""
    try:
        return _queue().get(partition=training_run_id, timeout=timeout)
    except Exception:
        time.sleep(timeout)
        return None
