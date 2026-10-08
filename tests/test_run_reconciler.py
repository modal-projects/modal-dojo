"""Tests for orphaned training-run reconciliation."""

from __future__ import annotations

import time

import pytest

from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.common.run_reconciler import (
    PRE_APP_TIMEOUT_SECONDS,
    QUEUED_STAGE_TIMEOUT_SECONDS,
    _load_running_runs,
    reconcile_decision,
    reconcile_orphan_runs,
)
from modal_dojo.utils.metadata import MetadataStore, vol_put


def _run(
    *,
    training_run_id: str = "run-1",
    status: TrainingRunStatus = TrainingRunStatus.RUNNING,
    modal_app_id: str = "",
    updated_at: int = 0,
    started_at: int = 0,
    metadata: dict | None = None,
) -> TrainingRun:
    return TrainingRun(
        training_run_id=training_run_id,
        modal_app_id=modal_app_id,
        framework=Framework.SLIME,
        config={},
        status=status,
        created_at=updated_at or started_at or 1_700_000_000,
        started_at=started_at or updated_at or 1_700_000_000,
        updated_at=updated_at or started_at or 1_700_000_000,
        metadata=metadata,
    )


def test_completed_run_is_ignored():
    now = 2_000_000_000
    run = _run(status=TrainingRunStatus.COMPLETED)
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is False


def test_run_with_train_result_is_ignored():
    now = 2_000_000_000
    run = _run()
    decision = reconcile_decision(run, now=now, has_train_result=True, app_live=False)
    assert decision.should_terminalize is False


def test_stale_pre_app_run_without_modal_app_id_is_cancelled():
    now = 2_000_000_000
    run = _run(
        modal_app_id="",
        updated_at=now - PRE_APP_TIMEOUT_SECONDS - 60,
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is True
    assert decision.reason == "stale_pre_active_no_modal_app"


def test_pre_app_orphan_uses_created_at_not_updated_at():
    now = 2_000_000_000
    created = now - PRE_APP_TIMEOUT_SECONDS - 60
    run = TrainingRun(
        training_run_id="old-pre-app",
        modal_app_id="",
        framework=Framework.SLIME,
        config={},
        status=TrainingRunStatus.RUNNING,
        created_at=created,
        started_at=created,
        updated_at=now - 60,
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is True
    assert decision.reason == "stale_pre_active_no_modal_app"


def test_recent_pre_app_run_stays_running():
    now = 2_000_000_000
    recent = now - 30 * 60
    run = TrainingRun(
        training_run_id="recent-pre-app",
        modal_app_id="",
        framework=Framework.SLIME,
        config={},
        status=TrainingRunStatus.RUNNING,
        created_at=recent,
        started_at=recent,
        updated_at=recent,
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is False


def test_unreachable_modal_app_is_cancelled_after_pre_app_timeout():
    now = 2_000_000_000
    stale = now - PRE_APP_TIMEOUT_SECONDS - 60
    run = _run(
        modal_app_id="ap-missing",
        updated_at=stale,
        metadata={
            "framework_progress": {
                "phase": "download_model",
                "is_active": False,
                "updated_at": stale,
            }
        },
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is True
    assert decision.reason == "stale_modal_app_unreachable"


def test_dead_modal_app_without_train_result_is_cancelled():
    now = 2_000_000_000
    run = _run(modal_app_id="ap-123", updated_at=now - 120)
    decision = reconcile_decision(
        run,
        now=now,
        has_train_result=False,
        app_live=False,
        modal_app_state=99,
    )
    assert decision.should_terminalize is True
    assert decision.reason == "stale_modal_app_terminated"
    assert decision.modal_app_state == 99


def test_live_modal_app_stays_running():
    now = 2_000_000_000
    run = _run(modal_app_id="ap-123", updated_at=now - 120)
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=True)
    assert decision.should_terminalize is False


def test_unknown_modal_app_state_does_not_cancel_recent_run():
    now = 2_000_000_000
    run = _run(modal_app_id="ap-123", updated_at=now - 120)
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is False


def test_stale_queued_stage_is_cancelled():
    now = 2_000_000_000
    run = _run(
        modal_app_id="ap-queued",
        updated_at=now - 60,
        metadata={
            "framework_progress": {
                "phase": "download_model",
                "is_active": False,
                "updated_at": now - QUEUED_STAGE_TIMEOUT_SECONDS - 60,
            }
        },
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=True)
    assert decision.should_terminalize is True
    assert decision.reason == "stale_queued_stage"


def test_active_queued_stage_is_not_cancelled():
    now = 2_000_000_000
    run = _run(
        modal_app_id="ap-queued",
        metadata={
            "framework_progress": {
                "phase": "download_model",
                "is_active": True,
                "updated_at": now - QUEUED_STAGE_TIMEOUT_SECONDS - 60,
            }
        },
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=True)
    assert decision.should_terminalize is False


def test_very_stale_run_with_dead_modal_app_is_cancelled():
    now = 2_000_000_000
    run = _run(
        modal_app_id="ap-dead",
        updated_at=now - 24 * 3600 - 60,
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=False)
    assert decision.should_terminalize is True
    assert decision.reason == "stale_modal_app_terminated"


def test_very_stale_run_with_live_modal_app_is_not_cancelled():
    now = 2_000_000_000
    run = _run(
        modal_app_id="ap-live",
        updated_at=now - 24 * 3600 - 60,
    )
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=True)
    assert decision.should_terminalize is False


@pytest.mark.parametrize(
    "status",
    [
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.FAILED,
        TrainingRunStatus.STOPPED,
        TrainingRunStatus.CANCELLED,
    ],
)
def test_terminal_runs_are_ignored(status):
    now = 2_000_000_000
    run = _run(status=status, modal_app_id="")
    decision = reconcile_decision(run, now=now, has_train_result=False, app_live=None)
    assert decision.should_terminalize is False


def test_reconcile_orphan_runs_persists_cancelled_status(fake_volume, monkeypatch):
    now = int(time.time())
    stale_run = _run(
        training_run_id="orphan-1",
        modal_app_id="",
        updated_at=now - PRE_APP_TIMEOUT_SECONDS - 60,
        started_at=now - PRE_APP_TIMEOUT_SECONDS - 60,
    )
    monkeypatch.setattr(
        "modal_dojo.common.run_reconciler._load_running_runs",
        lambda: [stale_run],
    )

    results = reconcile_orphan_runs(
        now=now,
        get_lifecycle_state=lambda _app_id: None,
        has_train_result=lambda _run_id: False,
    )
    assert len(results) == 1
    assert results[0].reason == "stale_pre_active_no_modal_app"
    assert stale_run.status == TrainingRunStatus.CANCELLED
    assert stale_run.metadata["terminal_reason"] == "stale_pre_active_no_modal_app"
    assert stale_run.metadata["reconciled_at"] == now

    saved = TrainingRun.from_id("orphan-1")
    assert saved.status == TrainingRunStatus.CANCELLED


def test_reconcile_orphan_runs_skips_run_with_train_result(fake_volume):
    now = int(time.time())
    stale_run = _run(
        training_run_id="orphan-2",
        modal_app_id="ap-1",
        updated_at=now - PRE_APP_TIMEOUT_SECONDS - 60,
    )
    stale_run.save()
    vol_put(
        MetadataStore.TRAIN_RESULTS,
        "orphan-2",
        {"training_run_id": "orphan-2"},
    )

    results = reconcile_orphan_runs(
        now=now,
        get_lifecycle_state=lambda _app_id: 99,
        has_train_result=lambda run_id: run_id == "orphan-2",
    )
    assert results == []
    saved = TrainingRun.from_id("orphan-2")
    assert saved.status == TrainingRunStatus.RUNNING


def test_load_running_runs_reads_canonical_store_not_only_summary(fake_volume):
    now = int(time.time())
    created = now - PRE_APP_TIMEOUT_SECONDS - 60
    stale_run = TrainingRun(
        training_run_id="canonical-only",
        modal_app_id="",
        framework=Framework.SLIME,
        config={},
        status=TrainingRunStatus.RUNNING,
        created_at=created,
        started_at=created,
        updated_at=created,
    )
    vol_put(
        MetadataStore.TRAINING_RUNS,
        "canonical-only",
        stale_run.model_dump(mode="json"),
    )
    vol_put(
        MetadataStore.TRAINING_RUNS_SUMMARY,
        "summary",
        {"items": []},
    )

    loaded = _load_running_runs()
    assert {run.training_run_id for run in loaded} == {"canonical-only"}


def test_resolve_app_liveness_fetches_lifecycle_once():
    from modal_dojo.common.modal_lifecycle import resolve_app_liveness

    calls: list[str] = []

    def get_state(app_id: str) -> int:
        calls.append(app_id)
        return 99

    state, live = resolve_app_liveness("ap-123", get_lifecycle_state=get_state)

    assert calls == ["ap-123"]
    assert state == 99
    assert live is False
