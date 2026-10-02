from __future__ import annotations

import pytest

from modal_dojo.common.errors import DojoError
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import (
    TrainingRun,
    TrainingRunStatus,
    mark_training_attempt_started,
)
from modal_dojo.utils.metadata import MetadataStore, vol_put


def _run(status: TrainingRunStatus) -> TrainingRun:
    return TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
        status=status,
    )


def test_done_is_false_while_running(fake_volume):
    assert _run(TrainingRunStatus.RUNNING).done() is False


@pytest.mark.parametrize(
    "status",
    [
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.FAILED,
        TrainingRunStatus.STOPPED,
        TrainingRunStatus.CANCELLED,
    ],
)
def test_done_is_true_for_terminal_status(status: TrainingRunStatus, fake_volume):
    assert _run(status).done() is True


def test_done_sees_status_written_after_launch(fake_volume):
    live = _run(TrainingRunStatus.RUNNING)
    _run(TrainingRunStatus.COMPLETED).save()

    assert live.done() is True
    assert live.status is TrainingRunStatus.COMPLETED


def test_context_manager_stops_app(monkeypatch):
    stopped: list[str] = []

    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app_best_effort", stopped.append
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with run as entered:
        assert entered is run
        assert stopped == []

    assert stopped == ["ap-1"]


def test_context_manager_leaves_app_running_on_exception(monkeypatch):
    stopped: list[str] = []

    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app_best_effort", stopped.append
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with pytest.raises(KeyboardInterrupt):
        with run:
            raise KeyboardInterrupt

    assert stopped == []


def test_close_is_idempotent(monkeypatch):
    stopped: list[str] = []

    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app_best_effort", stopped.append
    )
    run = _run(TrainingRunStatus.COMPLETED)
    run.modal_app_id = "ap-1"

    run.close()
    run.close()

    assert stopped == ["ap-1"]


def test_done_does_not_stop_app(monkeypatch, fake_volume):
    stopped: list[str] = []

    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app_best_effort", stopped.append
    )
    run = _run(TrainingRunStatus.COMPLETED)
    run.modal_app_id = "ap-1"

    assert run.done() is True
    assert stopped == []


class _FinishedCall:
    def get(self, timeout=None):
        del timeout
        return {"app_name": "done"}


class _FailedCall:
    def get(self, timeout=None):
        del timeout
        raise RuntimeError("worker died")


class _PendingCall:
    def get(self, timeout=None):
        del timeout
        raise TimeoutError()


class _TimeoutCall:
    def get(self, timeout=None):
        del timeout
        raise TimeoutError("expired")


def test_done_is_true_when_function_call_succeeds(fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _FinishedCall()

    assert run.done() is True
    assert run.status is TrainingRunStatus.COMPLETED


def test_done_is_true_when_function_call_dies(fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _FailedCall()

    assert run.done() is True
    assert run.status is TrainingRunStatus.FAILED
    assert run.error == "worker died"


def test_done_keeps_terminal_metadata_when_function_call_dies(fake_volume):
    run = _run(TrainingRunStatus.STOPPED)
    run._function_call = _FailedCall()

    assert run.done() is True
    assert run.status is TrainingRunStatus.STOPPED


def test_done_is_false_while_function_call_is_pending(fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _PendingCall()

    assert run.done() is False
    assert run.status is TrainingRunStatus.RUNNING


def test_wait_timeout_does_not_mark_failed(fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _TimeoutCall()

    with pytest.raises(TimeoutError, match="expired"):
        run.wait(timeout=0.01)

    assert run.status is TrainingRunStatus.RUNNING
    assert run.error is None


def test_wait_all_closes_each_run_when_that_run_is_done(monkeypatch, fake_volume):
    stopped: list[str] = []

    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app_best_effort", stopped.append
    )
    sleeps: list[float] = []

    class _FlipCall:
        pending = True

        def get(self, timeout=None):
            del timeout
            if self.pending:
                raise TimeoutError()
            return {}

    first = _run(TrainingRunStatus.COMPLETED)
    first.training_run_id = "run-a"
    first.modal_app_id = "ap-a"
    second = _run(TrainingRunStatus.RUNNING)
    second.training_run_id = "run-b"
    second.modal_app_id = "ap-b"
    second._function_call = _FlipCall()

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        second._function_call.pending = False

    monkeypatch.setattr("modal_dojo.common.run.time.sleep", _sleep)

    finished = TrainingRun.wait_all([first, second], poll_interval=0.01)

    assert finished == [first, second]
    assert stopped == ["ap-a", "ap-b"]
    assert sleeps == [0.01]
    assert second.status is TrainingRunStatus.COMPLETED


def test_stop_stops_app_and_persists_record(monkeypatch, fake_volume):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: True
    )
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app",
        stopped.append,
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.started_at = 100
    run.save()

    assert run.stop() is True

    assert stopped == ["ap-1"]
    persisted = TrainingRun.from_id("run-1")
    assert persisted.status is TrainingRunStatus.STOPPED
    assert persisted.ended_at is not None
    assert persisted.completed_at is None
    assert persisted.metadata["terminal_reason"] == "stopped_by_user"
    assert persisted.done() is True
    assert run.ended_at == persisted.ended_at
    assert run.duration_seconds == persisted.duration_seconds is not None
    assert run.completed_at is None


def test_stop_keeps_stored_config_over_stale_handle(monkeypatch, fake_volume):
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: True
    )
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app", lambda app_id: None
    )
    handle = _run(TrainingRunStatus.RUNNING)
    handle.modal_app_id = "ap-1"
    handle.save()
    stored = TrainingRun.from_id("run-1")
    stored.config = {"wandb_run_id": "attempt-2"}
    stored.save()

    assert handle.stop() is True

    assert TrainingRun.from_id("run-1").config == {"wandb_run_id": "attempt-2"}


def test_stop_is_noop_for_terminal_run(monkeypatch, fake_volume):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app",
        stopped.append,
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.save()
    _run(TrainingRunStatus.COMPLETED).save()

    assert run.stop() is False
    assert stopped == []
    assert TrainingRun.from_id("run-1").status is TrainingRunStatus.COMPLETED


def test_stop_does_not_persist_when_app_stop_fails(monkeypatch, fake_volume):
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: True
    )

    def fail_stop(app_id: str) -> None:
        raise RuntimeError("modal is down")

    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", fail_stop)
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.save()

    with pytest.raises(RuntimeError, match="modal is down"):
        run.stop()

    assert run.status is TrainingRunStatus.RUNNING
    assert TrainingRun.from_id("run-1").status is TrainingRunStatus.RUNNING


def test_stop_refuses_run_without_modal_app(monkeypatch, fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run.save()

    with pytest.raises(DojoError, match="still launching"):
        run.stop()

    assert TrainingRun.from_id("run-1").status is TrainingRunStatus.RUNNING


def test_stop_reconciles_record_when_app_already_dead(monkeypatch, fake_volume):
    calls: list[str] = []
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: False
    )
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", calls.append)
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.started_at = 100
    run.save()

    assert run.stop() is True

    assert calls == []
    persisted = TrainingRun.from_id("run-1")
    assert persisted.status is TrainingRunStatus.STOPPED
    assert persisted.ended_at is not None
    assert persisted.metadata["terminal_reason"] == "stopped_by_user"


def test_stop_marks_completed_when_train_result_exists(monkeypatch, fake_volume):
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: False
    )
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.stop_app",
        lambda app_id: pytest.fail("stop_app should not run"),
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.started_at = 100
    run.save()
    vol_put(MetadataStore.TRAIN_RESULTS, "run-1", {"training_run_id": "run-1"})

    assert run.stop() is True

    persisted = TrainingRun.from_id("run-1")
    assert persisted.status is TrainingRunStatus.COMPLETED
    assert persisted.completed_at is not None
    assert persisted.metadata.get("last_attempt_status") == "completed"
    assert "terminal_reason" not in (persisted.metadata or {})


def test_stop_attempts_rpc_when_liveness_unknown(monkeypatch, fake_volume):
    calls: list[str] = []
    monkeypatch.setattr(
        "modal_dojo.common.modal_lifecycle.app_live_status", lambda app_id: None
    )
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", calls.append)
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"
    run.save()

    assert run.stop() is True
    assert calls == ["ap-1"]
    assert TrainingRun.from_id("run-1").status is TrainingRunStatus.STOPPED


def test_save_keeps_stored_stopped_status(fake_volume):
    stopped = _run(TrainingRunStatus.STOPPED)
    stopped.ended_at = 200
    stopped.metadata = {"terminal_reason": "stopped_by_user"}
    stopped.save()
    stale = _run(TrainingRunStatus.FAILED)
    stale.ended_at = 300
    stale.error_message = "boom"
    stale.metadata = {"terminal_reason": "failed"}

    stale.save()

    persisted = TrainingRun.from_id("run-1")
    assert persisted.status is TrainingRunStatus.STOPPED
    assert persisted.ended_at == 200
    assert persisted.error_message is None
    assert persisted.metadata["terminal_reason"] == "stopped_by_user"


def test_save_lets_retry_overwrite_stopped_status(fake_volume):
    stopped = _run(TrainingRunStatus.STOPPED)
    stopped.ended_at = 200
    stopped.metadata = {"terminal_reason": "stopped_by_user", "attempt_count": 1}
    stopped.save()

    retry = TrainingRun.from_id("run-1")
    mark_training_attempt_started(retry, started_at=300)
    retry.save()

    persisted = TrainingRun.from_id("run-1")
    assert persisted.status is TrainingRunStatus.RUNNING
    assert persisted.ended_at is None
    assert "terminal_reason" not in persisted.metadata
    assert persisted.metadata["attempt_count"] == 2
