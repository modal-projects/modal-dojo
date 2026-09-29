"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

import copy
import asyncio
import math
import inspect
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import uuid4

from modal.exception import Error, NotFoundError
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
)

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.status import FrameworkStatus, resolve_framework_status
from modal_training_gym.common.torch_dist_checkpoint import (
    TORCH_DIST_TRACKER_NAME,
    parse_torch_dist_iteration,
    parse_torch_dist_tracker,
)
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_list_prefix,
    vol_put,
    vol_put_with_summary,
)

if TYPE_CHECKING:
    from modal_training_gym.common.checkpoint import Checkpoint
    from modal_training_gym.common.dashboard_components import DashboardComponent
    from modal_training_gym.common.training_rollout import TrainingRolloutResult

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value
CHECKPOINT_LOCATION_METADATA_KEY = "checkpoint_location"


def run_update_keys(data: dict[str, Any]) -> dict[str, str]:
    run_id = data.get("training_run_id") or data.get("run_id")
    if not run_id:
        return {}
    attempt = (data.get("metadata") or {}).get("attempt_count", 0)
    return {
        field: f"{run_id}__{attempt}__{field}"
        for field in ("framework_progress", "latest_rollout")
    }


def merge_run_updates(
    data: dict[str, Any], updates: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    data = {**data, "metadata": dict(data.get("metadata") or {})}
    for field, key in run_update_keys(data).items():
        candidates = updates.get(key)
        if not candidates:
            continue
        existing = data["metadata"].get(field) or {}
        order = (
            ("updated_at",)
            if field == "framework_progress"
            else ("rollout_id", "created_at")
        )
        update = max(
            candidates,
            key=lambda item: tuple(item[field].get(k, 0) for k in order),
        )
        if tuple(existing.get(k, 0) for k in order) > tuple(
            update[field].get(k, 0) for k in order
        ):
            continue
        data["metadata"][field] = update[field]
        data["updated_at"] = max(data.get("updated_at") or 0, update["updated_at"])
        if field == "framework_progress":
            data["framework_status"] = update["framework_status"]
    return data


class FrameworkStatusUpdate(BaseModel):
    """Body of ``POST /api/framework-status``.

    Reporters (``common/status_reporter.py``, slime's ``phase_reporting``) post
    more keys than the dashboard tracks (``app_name``, ``metrics``, …); those
    extras are ignored. Progress values come from loosely-typed framework args,
    so anything non-numeric or negative reads as "not provided".
    """

    training_run_id: str
    phase: str
    is_active: bool | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_unit: str | None = None
    rollout_id: int | None = None
    step_id: int | None = None
    step_event: str = ""
    # Client-side timestamp of the event (time.time() in the reporting
    # process); step timings use it so queue/network latency doesn't skew them.
    event_ts: float | None = None

    @field_validator(
        "progress_current", "progress_total", "rollout_id", "step_id", mode="before"
    )
    @classmethod
    def _non_negative_int_or_none(cls, value: object) -> int | None:
        if not isinstance(value, (int, float, str)):
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None


class TrainingRunStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRun(BaseModel):
    """A launched training run that can be inspected, awaited, or loaded by ID.

    Attributes:
        training_run_id: Stable id for this run in the metadata volume.
        modal_app_id: Modal app id after the run is spawned.
        modal_app_url: Dashboard URL for that Modal app.
        framework: Training framework that executed the run.
        config: Serialized train config captured at launch.
        dataset_id: Dataset id materialized for this run.
        deployment_id: Linked deployment id, when the run served a model.
        status: Gym-level run status.
        framework_status: Latest phase reported by the framework.
        created_at: Unix time the record was created.
        started_at: Unix time training started.
        ended_at: Unix time the process exited.
        completed_at: Unix time the run reached a terminal success.
        updated_at: Unix time the record was last written.
        duration_seconds: Wall time from start to end, when known.
        step_times: Per-step timings, when present.
        substep_times: Per-substep timings, when present.
        error_message: Failure text. Empty while running or after success.
        metadata: Extra keys such as sweep ``group_id``.
        function_call_id: Modal FunctionCall id used to wait on the spawned run.
        app_name: Modal app name used as the default checkpoint volume prefix.
        source_model: Model used for training.
        metrics: Final scalar metrics written at the end of the run.
    """

    training_run_id: str
    modal_app_id: str = ""
    modal_app_url: str = ""
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    status: TrainingRunStatus = TrainingRunStatus.RUNNING
    framework_status: FrameworkStatus | None = None
    created_at: int = 0
    started_at: int = 0
    ended_at: int | None = None
    completed_at: int | None = None
    updated_at: int = 0
    duration_seconds: int | None = None
    # TODO(joy): Remove with the legacy timing runs; retained for old runs and
    # serialized validation-harness results to keep rendering them.
    step_times: dict[str, dict[str, int | None]] | None = None
    # TODO(joy): Remove with the legacy timing runs; retained for old runs and
    # serialized validation-harness results to keep rendering them.
    substep_times: dict[str, dict[str, dict[str, float | int | bool | None]]] | None = (
        None
    )
    # Terminal failure message (Ray driver error / exception) for a failed run,
    # so the cause is queryable from the record and shown on the dashboard even
    # after logs roll off. None while running / on success.
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    # Handle to the spawned ``app.train`` Modal FunctionCall so a launched run
    # can be waited on (see ``result()`` / ``__await__``). Empty until the run
    # is actually spawned by ``TrainConfig.launch()``.
    function_call_id: str = ""
    app_name: str = ""
    source_model: Any = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    # Runtime-only handles attached by ``TrainConfig.launch()``; never persisted.
    _function_call: Any = PrivateAttr(default=None)
    _status_display: Any = PrivateAttr(default=None)
    _metadata_removed_keys: set[str] = PrivateAttr(default_factory=set)
    _metadata_loaded_keys: set[str] | None = PrivateAttr(default=None)
    _dashboard_component_updates: set[str] = PrivateAttr(default_factory=set)
    _closed: bool = PrivateAttr(default=False)

    @field_serializer("source_model")
    def _serialize_source_model(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return {
                "model_name": value.get("model_name", ""),
                "model_path": value.get("model_path"),
            }
        if not isinstance(value, ModelConfig):
            raise ValueError("source_model is not a ModelConfig or dict.")
        return {
            "model_name": value.model_name,
            "model_path": value.model_path,
        }

    @computed_field
    @property
    def group_id(self) -> str | None:
        """The sweep ID stored in ``metadata``."""
        meta = self.metadata or {}
        gid = meta.get("group_id")
        if gid:
            return str(gid)
        tags = meta.get("group_tags")
        if isinstance(tags, dict) and tags.get("group_id"):
            return str(tags["group_id"])
        return None

    # ── Launch-handle behavior (waiting on the spawned run) ──────────────────

    @property
    def function_call(self) -> Any:
        if self._function_call is not None:
            return self._function_call
        import modal

        return modal.FunctionCall.from_id(self.function_call_id)

    def _reload(self) -> None:
        try:
            stored = TrainingRun.from_id(self.training_run_id)
        except (KeyError, NotFoundError):
            return
        self.status = stored.status
        self.metadata = stored.metadata
        self._metadata_loaded_keys = stored._metadata_loaded_keys
        self.app_name = stored.app_name
        self.source_model = stored.source_model
        self.metrics = stored.metrics
        self.error_message = stored.error_message

    def _function_call_outcome(self) -> tuple[bool, BaseException | None]:
        if self._function_call is None and not self.function_call_id:
            return False, None
        try:
            call = self.function_call
        except Exception:
            return False, None
        try:
            call.get(timeout=0)
            return True, None
        except TimeoutError:
            return False, None
        except BaseException as exc:
            return True, exc

    def checkpoints(self) -> list["Checkpoint"]:
        """Snapshot of committed megatron ``iter_*`` directories.

        Miles LoRA adapters are not listed. Serving needs
        ``convert_megatron_checkpoint_to_hf``.
        """
        from modal_training_gym.common.checkpoint import _list_checkpoints

        self._reload()
        location = checkpoint_location(self)
        if location is None:
            return []
        directory, volume, mount = location
        return _list_checkpoints(
            directory,
            volume,
            mount,
            training_run_id=self.training_run_id,
            app_name=self.app_name,
        )

    def latest_checkpoint(self) -> "Checkpoint | None":
        checkpoints = self.checkpoints()
        return checkpoints[-1] if checkpoints else None

    @property
    def checkpoint_dir(self) -> str:
        location = checkpoint_location(self)
        return location[0] if location else ""

    @property
    def error(self) -> str | None:
        return self.error_message

    @property
    def model(self) -> "ModelConfig":
        """Build a ``ModelConfig`` whose path targets the latest megatron checkpoint."""
        self._reload()
        raw = self.source_model
        if raw is None:
            raise ValueError("No source_model on this TrainingRun.")
        if isinstance(raw, ModelConfig):
            model = copy.copy(raw)
        elif isinstance(raw, dict):
            model = ModelConfig(**raw)
        else:
            raise ValueError("source_model is not a ModelConfig or dict.")
        checkpoint = self.latest_checkpoint()
        if checkpoint is not None:
            model.model_path = checkpoint.path
        elif self.checkpoint_dir:
            model.model_path = self.checkpoint_dir
        return model

    def done(self) -> bool:
        """True if status is not RUNNING or the FunctionCall has finished.

        Does not stop the Modal app.
        """
        self._reload()
        if self.status is not TrainingRunStatus.RUNNING:
            return True
        finished, exc = self._function_call_outcome()
        if not finished:
            return False
        self.status = (
            TrainingRunStatus.FAILED if exc is not None else TrainingRunStatus.COMPLETED
        )
        if exc is not None:
            self.error_message = self.error_message or str(exc)
        return True

    def wait(self, *, timeout: float | None = None) -> "TrainingRun":
        """Block until this run is done. Does not stop the Modal app."""
        from modal_training_gym.common.status_reporter import (
            flush as flush_status_reporter,
        )

        if self._status_display is not None:
            self._status_display.start_polling(self.training_run_id)
        try:
            if self._function_call is not None or self.function_call_id:
                try:
                    payload = self.function_call.get(timeout=timeout)
                except TimeoutError as exc:
                    if not str(exc):
                        exc.args = (
                            f"Timed out after {timeout}s waiting for "
                            f"training_run_id={self.training_run_id}",
                        )
                    exc.training_run_id = self.training_run_id
                    raise
                except BaseException as exc:
                    message = str(exc)
                    if self.training_run_id not in message:
                        try:
                            exc.args = (
                                f"{message} (training_run_id={self.training_run_id})",
                                *exc.args[1:],
                            )
                        except (AttributeError, TypeError):
                            pass
                    try:
                        exc.training_run_id = self.training_run_id
                    except AttributeError:
                        pass
                    self._reload()
                    if self.status is TrainingRunStatus.RUNNING:
                        self.status = TrainingRunStatus.FAILED
                        self.error_message = self.error_message or str(exc)
                    raise
                self._reload()
                if isinstance(payload, dict):
                    if payload.get("app_name"):
                        self.app_name = str(payload["app_name"])
                    if "model_config" in payload:
                        self.source_model = payload["model_config"]
                    metrics = payload.get("metrics")
                    if isinstance(metrics, dict):
                        self.metrics = metrics
                return self
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self.done():
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for training_run_id={self.training_run_id}"
                    )
                time.sleep(1.0)
            return self
        finally:
            if self._status_display is not None:
                self._status_display.stop_polling()
            flush_status_reporter(timeout_seconds=2.0)

    def result(
        self,
        *,
        timeout: float | None = None,
        stop_app_on_success: bool = True,
    ) -> "TrainingRun":
        self.wait(timeout=timeout)
        if stop_app_on_success:
            self.close()
        print(f"Training complete: {self.training_run_id}")
        return self

    @classmethod
    def wait_all(
        cls,
        runs: Sequence["TrainingRun"],
        *,
        poll_interval: float = 30,
    ) -> list["TrainingRun"]:
        """Wait for every run. Close each run when that run is done."""
        pending = list(runs)
        while pending:
            still_running: list[TrainingRun] = []
            for run in pending:
                if run.done():
                    run.close()
                else:
                    still_running.append(run)
            pending = still_running
            if pending:
                time.sleep(poll_interval)
        return list(runs)

    def close(self) -> None:
        """Stop the detached Modal app. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        from modal_training_gym.common.modal_lifecycle import stop_app

        stop_app(self.modal_app_id)

    def __enter__(self) -> "TrainingRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()

    def __await__(self):
        import asyncio

        async def _wait() -> TrainingRun:
            return await asyncio.to_thread(self.result)

        return _wait().__await__()

    def _summary_sort_key(self, item: dict[str, Any]) -> tuple[int, str]:
        return (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        )

    def apply_framework_status(
        self, update: FrameworkStatusUpdate
    ) -> FrameworkStatus | None:
        """Apply a framework status update without saving the run.

        Returns:
            The resolved status, or ``None`` for an invalid phase.
        """
        status = resolve_framework_status(update.phase, str(self.framework.value))
        if status is None:
            return None

        self.framework_status = status
        progress: dict[str, Any] = {
            "phase": status.value,
            "updated_at": time.time(),
        }
        # is_active: True = stage is actually running on hardware; False =
        # we've marked the stage but it's queuing for a GPU. Sent by the
        # orchestration code in common/train.py (queue=False) and by the
        # Modal function itself when its body starts executing (active=True).
        if update.is_active is not None:
            progress["is_active"] = update.is_active
        for key, value in (
            ("current", update.progress_current),
            ("total", update.progress_total),
            ("unit", update.progress_unit),
            ("rollout_id", update.rollout_id),
            ("step_id", update.step_id),
        ):
            if value is not None:
                progress[key] = value

        metadata = dict(self.metadata or {})
        existing_progress = metadata.get("framework_progress")
        if isinstance(existing_progress, dict):
            # Drop the existing is_active when we get a fresh transition into
            # a different phase — it shouldn't bleed across stage changes.
            if existing_progress.get("phase") != progress["phase"]:
                existing_progress = {
                    k: v for k, v in existing_progress.items() if k != "is_active"
                }
            progress = {**existing_progress, **progress}
            for key in ("current", "rollout_id"):
                incoming = progress.get(key)
                existing = existing_progress.get(key)
                if incoming is None or existing is None:
                    continue
                try:
                    progress[key] = max(int(existing), int(incoming))
                except (TypeError, ValueError):
                    pass
        metadata["framework_progress"] = progress
        self.metadata = metadata

        return status

    def record_latest_rollout(self, rollout: TrainingRolloutResult) -> None:
        """Store the saved rollout summary in this run's metadata."""
        metadata = dict(self.metadata or {})
        metadata["latest_rollout"] = {
            "rollout_id": rollout.rollout_id,
            "mean": rollout.mean,
            "total": rollout.total,
            "created_at": rollout.created_at,
        }
        self.metadata = metadata

    def add_dashboard_component(
        self,
        *,
        name: str,
        component_type: "DashboardComponent | str",
        from_path: str | os.PathLike[str],
        replace: bool = False,
    ) -> dict[str, Any]:
        """Attach a local dashboard component to this run.

        The source is uploaded to the shared dashboard-overlay Volume and the
        resulting immutable manifest is associated with this run's metadata.
        This operation is independent of the training job and may be called
        after a run has started or completed.
        """
        from modal_training_gym.common.dashboard_components import (
            read_dashboard_component,
            store_dashboard_component,
        )

        metadata = dict(self.metadata or {})
        components = metadata.get("dashboard_components")
        components = dict(components) if isinstance(components, dict) else {}
        previous = components.get(name)
        if previous is not None and not replace:
            previous_hash = (
                previous.get("sha256") if isinstance(previous, dict) else None
            )
            _, _, digest = read_dashboard_component(
                component_type=component_type, from_path=from_path
            )
            if previous_hash != digest:
                raise ValueError(
                    f"dashboard component {name!r} is already attached to run "
                    f"{self.training_run_id!r}; pass replace=True to replace it"
                )

        manifest = store_dashboard_component(
            name=name,
            component_type=component_type,
            from_path=from_path,
            training_run_id=self.training_run_id,
        )
        # Re-insert so dict order reflects attachment order; the dashboard
        # resolves a component type to its most recently attached entry.
        components.pop(name, None)
        components[name] = manifest
        metadata["dashboard_components"] = components
        self.metadata = metadata
        self._dashboard_component_updates.add(name)
        self.save()
        return manifest

    def _touch(self) -> None:
        self.updated_at = int(time.time())

    async def _save_dashboard_update(
        self, field: Literal["framework_progress", "latest_rollout"]
    ) -> None:
        data = self.model_dump(mode="json")
        payload = {field: data["metadata"][field], "updated_at": int(time.time())}
        if field == "framework_progress":
            payload["framework_status"] = data["framework_status"]
        key = f"{run_update_keys(data)[field]}__{uuid4().hex}"
        await vol_put(
            MetadataStore.TRAINING_RUN_UPDATES,
            key,
            payload,
            is_async=True,
        )

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch()

        def payload_with_stored_metadata(stored: object) -> dict[str, Any]:
            payload = self.model_dump(mode="json")
            stored_metadata = (
                stored.get("metadata") if isinstance(stored, dict) else None
            )
            current_metadata = payload.get("metadata")
            stored_metadata = (
                stored_metadata if isinstance(stored_metadata, dict) else {}
            )
            current_metadata = (
                current_metadata if isinstance(current_metadata, dict) else {}
            )
            loaded_keys = self._metadata_loaded_keys or set()
            removed_keys = (loaded_keys - current_metadata.keys()) | (
                self._metadata_removed_keys
            )
            merged_metadata = {
                key: value
                for key, value in stored_metadata.items()
                if key not in removed_keys
            }
            merged_metadata.update(
                {key: value for key, value in current_metadata.items()}
            )
            stored_latest = stored_metadata.get("latest_rollout")
            current_latest = current_metadata.get("latest_rollout")
            if isinstance(stored_latest, dict) and isinstance(current_latest, dict):

                def rollout_key(value: dict[str, Any]) -> tuple[float, float]:
                    try:
                        rollout_id = float(value.get("rollout_id", -1))
                    except (TypeError, ValueError):
                        rollout_id = -1
                    try:
                        created_at = float(value.get("created_at", 0) or 0)
                    except (TypeError, ValueError):
                        created_at = 0
                    return rollout_id, created_at

                stored_key = rollout_key(stored_latest)
                current_key = rollout_key(current_latest)
                merged_metadata["latest_rollout"] = (
                    current_latest if current_key >= stored_key else stored_latest
                )
            elif isinstance(stored_latest, dict):
                merged_metadata["latest_rollout"] = stored_latest
            stored_progress = stored_metadata.get("framework_progress")
            current_progress = current_metadata.get("framework_progress")
            if isinstance(stored_progress, dict) and isinstance(current_progress, dict):
                try:
                    stored_updated_at = int(stored_progress.get("updated_at", 0) or 0)
                except (TypeError, ValueError):
                    stored_updated_at = 0
                try:
                    current_updated_at = int(current_progress.get("updated_at", 0) or 0)
                except (TypeError, ValueError):
                    current_updated_at = 0
                if stored_updated_at > current_updated_at:
                    merged_metadata["framework_progress"] = stored_progress
            elif isinstance(stored_progress, dict):
                merged_metadata["framework_progress"] = stored_progress
            # Components are attached by name from independent handles (often
            # after launch). The stored map is authoritative; only the names
            # this handle attached since its last save are written over it,
            # appended so insertion order stays attachment order (the
            # dashboard picks the last matching entry). A handle that merely
            # carries a stale copy of the map can neither drop nor revert
            # another handle's attachments.
            stored_components = stored_metadata.get("dashboard_components")
            current_components = current_metadata.get("dashboard_components")
            if isinstance(stored_components, dict) and isinstance(
                current_components, dict
            ):
                merged_components = dict(stored_components)
                for name in self._dashboard_component_updates:
                    if name not in current_components:
                        continue
                    merged_components.pop(name, None)
                    merged_components[name] = current_components[name]
                merged_metadata["dashboard_components"] = merged_components
            payload["metadata"] = merged_metadata
            return payload

        def write(payload: dict[str, Any]) -> None | Awaitable[None]:
            return vol_put_with_summary(
                MetadataStore.TRAINING_RUNS,
                self.training_run_id,
                payload,
                summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
                item_id_key="training_run_id",
                sort_key=self._summary_sort_key,
                reverse=True,
                is_async=is_async,
            )

        try:
            stored = vol_get(
                MetadataStore.TRAINING_RUNS, self.training_run_id, is_async=is_async
            )
        except Exception:
            stored = None
        if not is_async:
            write(payload_with_stored_metadata(stored))
            self._dashboard_component_updates.clear()
            return None

        async def _save_async() -> None:
            stored_data = None
            if inspect.isawaitable(stored):
                try:
                    stored_data = await stored
                except Exception:
                    stored_data = None
            result = write(payload_with_stored_metadata(stored_data))
            if result is not None:
                await result
            self._dashboard_component_updates.clear()

        return _save_async()

    @classmethod
    @overload
    def from_id(
        cls, run_id: str, *, is_async: Literal[True]
    ) -> Awaitable[TrainingRun]: ...

    @classmethod
    @overload
    def from_id(
        cls, run_id: str, *, is_async: Literal[False] = False
    ) -> TrainingRun: ...

    @classmethod
    def from_id(
        cls, run_id: str, *, is_async: bool = False
    ) -> TrainingRun | Awaitable[TrainingRun]:
        data = vol_get(MetadataStore.TRAINING_RUNS, run_id, is_async=is_async)
        if is_async:

            async def _run() -> TrainingRun:
                stored = await data

                async def read(key: str) -> tuple[str, list[dict[str, Any]]]:
                    try:
                        return key, await asyncio.to_thread(
                            vol_list_prefix, MetadataStore.TRAINING_RUN_UPDATES, key
                        )
                    except (KeyError, Error):
                        return key, []

                updates = dict(
                    await asyncio.gather(
                        *(read(key) for key in run_update_keys(stored).values())
                    )
                )
                return cls.from_stored_data(merge_run_updates(stored, updates))

            return _run()
        updates = {}
        for key in run_update_keys(data).values():
            try:
                updates[key] = vol_list_prefix(MetadataStore.TRAINING_RUN_UPDATES, key)
            except (KeyError, Error):
                pass
        return cls.from_stored_data(merge_run_updates(data, updates))

    @classmethod
    def from_stored_data(cls, data: object) -> TrainingRun:
        """Build a run from persisted data and snapshot its metadata keys."""
        run = cls.model_validate(data)
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict):
            run._metadata_loaded_keys = set(metadata)
        return run


def checkpoint_location(
    run: TrainingRun,
) -> tuple[str, str, str] | None:
    loc = (run.metadata or {}).get(CHECKPOINT_LOCATION_METADATA_KEY)
    if not isinstance(loc, dict):
        return None
    directory = loc.get("checkpoint_dir")
    volume = loc.get("checkpoints_volume_name")
    mount = loc.get("checkpoints_mount_path")
    if not (
        isinstance(directory, str) and directory and isinstance(volume, str) and volume
    ):
        return None
    mount_path = mount if isinstance(mount, str) and mount else "/checkpoints"
    return directory, volume, mount_path


def set_checkpoint_location(
    run: TrainingRun,
    *,
    checkpoint_dir: str,
    checkpoints_volume_name: str,
    checkpoints_mount_path: str,
) -> None:
    metadata = dict(run.metadata or {})
    metadata[CHECKPOINT_LOCATION_METADATA_KEY] = {
        "checkpoint_dir": checkpoint_dir,
        "checkpoints_volume_name": checkpoints_volume_name,
        "checkpoints_mount_path": checkpoints_mount_path,
    }
    run.metadata = metadata


def _resume_checkpoint(path: str, name: str, iteration: int | None) -> dict[str, Any]:
    return {
        "resume_checkpoint_path": path,
        "resume_checkpoint_name": name,
        "resume_from_iteration": iteration,
    }


def has_torch_dist_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> bool:
    return torch_dist_resume_checkpoint(save_path, is_complete=is_complete) is not None


def torch_dist_resume_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> dict[str, Any] | None:
    if not os.path.isdir(save_path):
        return None

    is_complete = is_complete or os.path.isdir
    tracker_path = os.path.join(save_path, TORCH_DIST_TRACKER_NAME)
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                tracker = f.read()
        except OSError:
            tracker = ""
        if tracker.strip() == "release":
            path = os.path.join(save_path, "release")
            return (
                _resume_checkpoint(path, "release", None) if is_complete(path) else None
            )
        iteration = parse_torch_dist_tracker(tracker)
        if iteration is not None:
            name = f"iter_{iteration:07d}"
            path = os.path.join(save_path, name)
            return (
                _resume_checkpoint(path, name, iteration) if is_complete(path) else None
            )

    try:
        candidates: list[tuple[int, str, str]] = []
        release_path = ""
        for entry in os.scandir(save_path):
            if not entry.is_dir() or not is_complete(entry.path):
                continue
            if entry.name == "release":
                release_path = entry.path
            elif entry.name.startswith("iter_"):
                iteration = parse_torch_dist_iteration(entry.name)
                if iteration is None:
                    continue
                candidates.append((iteration, entry.name, entry.path))
    except OSError:
        return None

    if candidates:
        iteration, name, path = max(candidates)
        return _resume_checkpoint(path, name, iteration)
    if release_path:
        return _resume_checkpoint(release_path, "release", None)
    return None


def run_scoped_save_root(save_root: str, training_run_id: str) -> str:
    save_root = str(save_root).rstrip("/") or "/"
    if os.path.basename(save_root) == training_run_id:
        return save_root
    return os.path.join(save_root, training_run_id)


def mark_training_attempt_started(
    run: TrainingRun,
    *,
    started_at: int,
) -> int:
    metadata = dict(run.metadata or {})
    try:
        attempt_count = int(metadata.get("attempt_count") or 0) + 1
    except (TypeError, ValueError):
        attempt_count = 1

    metadata["attempt_count"] = attempt_count
    metadata["last_attempt_started_at"] = started_at
    metadata["last_attempt_status"] = "running"
    raw_attempt_starts = metadata.get("attempt_starts")
    attempt_starts: set[int] = set()
    if isinstance(raw_attempt_starts, list):
        for value in raw_attempt_starts:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(parsed) and parsed.is_integer():
                attempt_starts.add(int(parsed))
    attempt_starts.add(started_at)
    metadata["attempt_starts"] = sorted(attempt_starts)[-50:]
    metadata.pop("terminal_reason", None)
    run._metadata_removed_keys.add("terminal_reason")

    run.status = TrainingRunStatus.RUNNING
    run.ended_at = None
    run.completed_at = None
    run.duration_seconds = None
    run.error_message = None
    run.metadata = metadata
    return attempt_count


def mark_training_attempt_finished(
    run: TrainingRun,
    *,
    status: str,
    ended_at: int,
) -> None:
    metadata = dict(run.metadata or {})
    metadata["last_attempt_status"] = status
    metadata["last_attempt_ended_at"] = ended_at
    run.metadata = metadata


def record_resume_checkpoint(
    run: TrainingRun,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    metadata = dict(run.metadata or {})
    if resume_checkpoint is None:
        metadata["resumed_from_checkpoint"] = False
        for key in (
            "resume_checkpoint_path",
            "resume_checkpoint_name",
            "resume_from_iteration",
        ):
            metadata.pop(key, None)
            run._metadata_removed_keys.add(key)
        progress = metadata.get("framework_progress")
        if isinstance(progress, dict):
            progress = dict(progress)
            for key in ("current", "rollout_id", "step_id"):
                progress.pop(key, None)
            progress["updated_at"] = int(time.time())
            metadata["framework_progress"] = progress
    else:
        metadata["resumed_from_checkpoint"] = True
        metadata.update(resume_checkpoint)
        run._metadata_removed_keys.difference_update(resume_checkpoint)
        progress = metadata.get("framework_progress")
        if isinstance(progress, dict):
            progress = dict(progress)
            try:
                raw_resume_iteration = resume_checkpoint.get("resume_from_iteration")
                resume_iteration = (
                    None if raw_resume_iteration is None else int(raw_resume_iteration)
                )
            except (TypeError, ValueError):
                resume_iteration = None
            if resume_iteration is None:
                for key in ("current", "rollout_id", "step_id"):
                    progress.pop(key, None)
            else:
                # Checkpoint iterations are zero-based completed rollouts.
                # Progress.current is one-based, so the completed-step floor
                # is resume_iteration + 1; zero-based lane identities use the
                # checkpoint iteration itself.
                progress["current"] = resume_iteration + 1
                progress["rollout_id"] = max(0, resume_iteration)
                progress["step_id"] = max(0, resume_iteration)
            progress["updated_at"] = int(time.time())
            metadata["framework_progress"] = progress
    run.metadata = metadata


def metric_run_id_for_attempt(training_run_id: str, attempt_count: int) -> str:
    return (
        training_run_id if attempt_count <= 1 else f"{training_run_id}-a{attempt_count}"
    )


def record_metric_attempt(
    run: TrainingRun,
    *,
    provider: str,
    entity: str,
    project: str,
    group: str,
    run_id: str,
    url: str = "",
    attempt_count: int,
) -> None:
    if not run_id:
        return

    metadata = dict(run.metadata or {})
    raw_attempts = metadata.get("metric_attempts") or metadata.get("wandb_attempts")
    attempts = raw_attempts if isinstance(raw_attempts, list) else []
    attempt = {
        "attempt": attempt_count,
        "provider": provider,
        "entity": entity,
        "project": project,
        "group": group,
        "run_id": run_id,
        "url": url,
    }
    attempts = [
        existing
        for existing in attempts
        if not (
            isinstance(existing, dict)
            and (
                existing.get("attempt") == attempt_count
                or existing.get("run_id") == run_id
            )
        )
    ]
    attempts.append(attempt)
    metadata["metric_attempts"] = sorted(
        attempts,
        key=lambda item: int(item.get("attempt") or 0) if isinstance(item, dict) else 0,
    )
    metadata["metric_latest_run_id"] = run_id
    metadata["metric_provider"] = provider
    run.metadata = metadata
