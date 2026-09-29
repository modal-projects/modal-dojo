"""Reconcile orphaned training runs stuck in ``running`` status."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modal_dojo.common.modal_lifecycle import resolve_app_liveness
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_get_summary_items_healed,
    vol_list,
)

PRE_APP_TIMEOUT_SECONDS = 90 * 60
QUEUED_STAGE_TIMEOUT_SECONDS = 4 * 3600
# Backstop for an active run whose Modal app can't be queried (persistent API
# failures returning ``app_live is None``): without this, a run that made it
# past the queued stages could sit in ``running`` forever. A dead app
# (``app_live is False``) is caught immediately by ``stale_modal_app_terminated``;
# this only covers the never-resolves-either-way case, so the window is long.
STALE_ACTIVE_TIMEOUT_SECONDS = 24 * 3600

QUEUEABLE_STAGES = frozenset({"initializing", "download_model", "convert_model"})


@dataclass(frozen=True)
class ReconcileDecision:
    should_terminalize: bool
    reason: str | None = None
    modal_app_state: int | None = None


@dataclass(frozen=True)
class ReconcileResult:
    training_run_id: str
    reason: str
    previous_status: str


def _has_modal_app(run: TrainingRun) -> bool:
    return bool(str(run.modal_app_id or "").strip())


def _run_created_at(run: TrainingRun) -> int:
    return int(run.created_at or run.started_at or 0)


def _run_last_activity(run: TrainingRun) -> int:
    return int(run.updated_at or run.started_at or run.created_at or 0)


def _pre_app_anchor(run: TrainingRun) -> int:
    # ``updated_at`` can be bumped by status posts even when no Modal app was
    # ever associated; ``created_at`` is the reliable signal for pre-app orphans.
    return _run_created_at(run) or _run_last_activity(run)


def _framework_progress(run: TrainingRun) -> dict[str, Any]:
    metadata = run.metadata or {}
    progress = metadata.get("framework_progress")
    return progress if isinstance(progress, dict) else {}


def _progress_updated_at(run: TrainingRun) -> int:
    progress = _framework_progress(run)
    return int(progress.get("updated_at") or _run_last_activity(run) or 0)


def _is_queued_pre_active(run: TrainingRun) -> bool:
    progress = _framework_progress(run)
    if progress.get("is_active") is not False:
        return False
    phase = str(progress.get("phase") or run.framework_status or "").strip().lower()
    if phase in QUEUEABLE_STAGES:
        return True
    if not _has_modal_app(run):
        return True
    return False


def reconcile_decision(
    run: TrainingRun,
    *,
    now: int,
    has_train_result: bool,
    app_live: bool | None,
    modal_app_state: int | None = None,
) -> ReconcileDecision:
    if run.status != TrainingRunStatus.RUNNING:
        return ReconcileDecision(False)
    if has_train_result:
        return ReconcileDecision(False)

    app_id = str(run.modal_app_id or "").strip()

    if app_id and app_live is False:
        return ReconcileDecision(
            True,
            reason="stale_modal_app_terminated",
            modal_app_state=modal_app_state,
        )

    if not app_id:
        pre_app_at = _pre_app_anchor(run)
        if pre_app_at <= 0 or now - pre_app_at >= PRE_APP_TIMEOUT_SECONDS:
            return ReconcileDecision(True, reason="stale_pre_active_no_modal_app")

    if app_id and app_live is None and _is_queued_pre_active(run):
        unreachable_at = _progress_updated_at(run) or _pre_app_anchor(run)
        if unreachable_at and now - unreachable_at >= PRE_APP_TIMEOUT_SECONDS:
            return ReconcileDecision(
                True,
                reason="stale_modal_app_unreachable",
                modal_app_state=modal_app_state,
            )

    if _is_queued_pre_active(run):
        queued_at = _progress_updated_at(run)
        if queued_at and now - queued_at >= QUEUED_STAGE_TIMEOUT_SECONDS:
            return ReconcileDecision(True, reason="stale_queued_stage")

    # Active run (past the queued stages) whose Modal app can't be queried and
    # has shown no activity for a very long window. The queued-stage and
    # unreachable-while-queued heuristics above only fire while still queued, so
    # this is the last-resort backstop for a run that resolves neither live nor
    # dead. Scoped to ``app_live is None`` so a live app is never reconciled.
    if app_id and app_live is None:
        last_activity = _run_last_activity(run)
        if last_activity and now - last_activity >= STALE_ACTIVE_TIMEOUT_SECONDS:
            return ReconcileDecision(
                True,
                reason="stale_running_no_update",
                modal_app_state=modal_app_state,
            )

    return ReconcileDecision(False)


def _default_has_train_result(training_run_id: str) -> bool:
    try:
        vol_get(MetadataStore.TRAIN_RESULTS, training_run_id)
        return True
    except KeyError:
        return False


def _parse_running_run(raw: dict[str, Any]) -> TrainingRun | None:
    if "training_run_id" not in raw and "run_id" in raw:
        raw = {**raw, "training_run_id": raw["run_id"]}
    try:
        run = TrainingRun.from_stored_data(raw)
    except Exception:
        return None
    if run.status != TrainingRunStatus.RUNNING:
        return None
    return run


def _load_running_runs() -> list[TrainingRun]:
    """Load running runs from canonical metadata, with healed summary as fallback."""
    runs_by_id: dict[str, TrainingRun] = {}

    for raw in vol_list(MetadataStore.TRAINING_RUNS):
        if not isinstance(raw, dict):
            continue
        run = _parse_running_run(raw)
        if run is not None:
            runs_by_id[run.training_run_id] = run

    for raw in vol_get_summary_items_healed(MetadataStore.TRAINING_RUNS_SUMMARY) or []:
        if not isinstance(raw, dict):
            continue
        run = _parse_running_run(raw)
        if run is not None:
            runs_by_id.setdefault(run.training_run_id, run)

    return list(runs_by_id.values())


def reconcile_orphan_runs(
    *,
    dry_run: bool = False,
    now: int | None = None,
    get_lifecycle_state: Callable[[str], int | None] | None = None,
    has_train_result: Callable[[str], bool] | None = None,
) -> list[ReconcileResult]:
    """Terminalize orphaned ``running`` runs. Returns reconciled run summaries."""
    now_ts = int(now if now is not None else time.time())
    has_result = has_train_result or _default_has_train_result

    results: list[ReconcileResult] = []
    for run in _load_running_runs():
        app_id = str(run.modal_app_id or "").strip()
        modal_app_state, app_live = resolve_app_liveness(
            app_id,
            get_lifecycle_state=get_lifecycle_state,
        )

        decision = reconcile_decision(
            run,
            now=now_ts,
            has_train_result=has_result(run.training_run_id),
            app_live=app_live,
            modal_app_state=modal_app_state,
        )
        if not decision.should_terminalize or not decision.reason:
            continue

        previous_status = run.status.value
        finished_at = now_ts
        metadata = dict(run.metadata or {})
        metadata["terminal_reason"] = decision.reason
        metadata["reconciled_at"] = finished_at
        if decision.modal_app_state is not None:
            metadata["last_known_modal_app_state"] = decision.modal_app_state

        run.status = TrainingRunStatus.CANCELLED
        run.metadata = metadata
        run.ended_at = finished_at
        if run.completed_at is None:
            run.completed_at = finished_at
        if run.started_at:
            run.duration_seconds = max(0, finished_at - run.started_at)

        result = ReconcileResult(
            training_run_id=run.training_run_id,
            reason=decision.reason,
            previous_status=previous_status,
        )
        if dry_run:
            results.append(result)
            continue

        try:
            run.save()
        except Exception as exc:
            print(f"WARNING: failed to reconcile {run.training_run_id}: {exc}")
            continue

        results.append(result)

    return results
