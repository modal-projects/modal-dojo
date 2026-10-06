from __future__ import annotations

import pytest
from modal.exception import ServiceError

from modal_dojo.common.framework import Framework
from modal_dojo.common.run import (
    TrainingRun,
    TrainingRunStatus,
    set_checkpoint_location,
)


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
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", stopped.append)
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with run as entered:
        assert entered is run
        assert stopped == []

    assert stopped == ["ap-1"]


def test_context_manager_leaves_app_running_on_exception(monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", stopped.append)
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with pytest.raises(KeyboardInterrupt):
        with run:
            raise KeyboardInterrupt

    assert stopped == []


def test_close_is_idempotent(monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", stopped.append)
    run = _run(TrainingRunStatus.COMPLETED)
    run.modal_app_id = "ap-1"

    run.close()
    run.close()

    assert stopped == ["ap-1"]


def test_done_does_not_stop_app(monkeypatch, fake_volume):
    stopped: list[str] = []
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", stopped.append)
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


def test_done_is_false_when_function_call_hits_transient_modal_error(fake_volume):
    class _FlakyCall:
        def get(self, timeout=None):
            del timeout
            raise ServiceError("unavailable")

    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _FlakyCall()

    assert run.done() is False
    assert run.status is TrainingRunStatus.RUNNING


def test_latest_checkpoint_retries_transient_modal_errors(monkeypatch, fake_volume):
    errors = [ServiceError("unavailable"), ServiceError("unavailable")]

    def _list_checkpoints(*args, **kwargs):
        if errors:
            raise errors.pop()
        return ["iter_0000001", "iter_0000002"]

    monkeypatch.setattr(
        "modal_dojo.common.checkpoint._list_checkpoints", _list_checkpoints
    )
    monkeypatch.setattr("modal_dojo.common.run.time.sleep", lambda seconds: None)
    run = _run(TrainingRunStatus.RUNNING)
    set_checkpoint_location(
        run,
        checkpoint_dir="/checkpoints/run-1",
        checkpoints_volume_name="ckpts",
        checkpoints_mount_path="/checkpoints",
    )

    assert run.latest_checkpoint() == "iter_0000002"


def test_wait_timeout_does_not_mark_failed(fake_volume):
    run = _run(TrainingRunStatus.RUNNING)
    run._function_call = _TimeoutCall()

    with pytest.raises(TimeoutError, match="expired"):
        run.wait(timeout=0.01)

    assert run.status is TrainingRunStatus.RUNNING
    assert run.error is None


def test_wait_all_closes_each_run_when_that_run_is_done(monkeypatch, fake_volume):
    stopped: list[str] = []
    monkeypatch.setattr("modal_dojo.common.modal_lifecycle.stop_app", stopped.append)
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
