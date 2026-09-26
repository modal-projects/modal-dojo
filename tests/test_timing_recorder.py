from __future__ import annotations

import sys
import time
from queue import Queue
from types import ModuleType

from modal_training_gym.common import reporting
from modal_training_gym.common import timing_recorder
from modal_training_gym.common.step_timing import RoleTimingRecord
from modal_training_gym.common.timing_recorder import RoleRecorder


def _configure(monkeypatch):
    monkeypatch.setattr(timing_recorder, "MIN_PUBLISH_INTERVAL_S", 0.0)
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "auto")
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dashboard.test")
    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-1")
    monkeypatch.setattr(timing_recorder, "_TIMING_MODE_CACHE", None)


def test_dist_down_rank_environment_elects_one_publisher(monkeypatch):
    dist = ModuleType("torch.distributed")
    dist.is_initialized = lambda: False
    torch = ModuleType("torch")
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    monkeypatch.setenv("RANK", "0")
    assert timing_recorder._lowest_rank_publishes() is True
    monkeypatch.setenv("RANK", "1")
    assert timing_recorder._lowest_rank_publishes() is False


def test_phase_accounting_and_invocations_are_cumulative(monkeypatch):
    _configure(monkeypatch)
    snapshots = []
    monkeypatch.setattr(
        reporting,
        "_enqueue_timing",
        lambda payload, *, final=False: snapshots.append((payload, final)),
    )

    recorder = RoleRecorder("actor", 7)
    with recorder.phase("forward_backward"):
        time.sleep(0.001)
    with recorder.phase("forward_backward"):
        time.sleep(0.001)

    phase = snapshots[-1][0]["phases"]["forward_backward"]
    assert phase["count"] == 2
    assert phase["busy_duration_s"] > 0
    assert len(phase["invocations"]) == 2
    assert (
        RoleTimingRecord.model_validate(snapshots[-1][0]).storage_key
        == "00000007__actor"
    )


def test_payload_bounds_are_applied_before_enqueue(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(timing_recorder, "MAX_PHASE_INVOCATIONS", 2)
    monkeypatch.setattr(timing_recorder, "MAX_TOTAL_INVOCATIONS", 2)
    snapshots = []
    monkeypatch.setattr(
        reporting,
        "_enqueue_timing",
        lambda payload, *, final=False: snapshots.append((payload, final)),
    )

    recorder = RoleRecorder("critic", 1)
    for _ in range(4):
        with recorder.phase("train"):
            pass

    payload = snapshots[-1][0]
    assert len(payload["phases"]["train"]["invocations"]) == 2
    assert RoleTimingRecord.model_validate(payload).phases["train"].count == 4


def test_off_mode_does_not_record_or_publish(monkeypatch):
    monkeypatch.setenv("TRAINING_GYM_SUBSTEP_TIMING", "off")
    monkeypatch.setattr(timing_recorder, "_TIMING_MODE_CACHE", None)
    snapshots = []
    monkeypatch.setattr(
        reporting,
        "_enqueue_timing",
        lambda payload, *, final=False: snapshots.append((payload, final)),
    )

    recorder = RoleRecorder("rollout", 0)
    with recorder.phase("generate_samples"):
        pass

    assert recorder.phases == {}
    assert snapshots == []


def test_sft_skips_rl_only_phases(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setenv("TRAINING_GYM_LOSS_TYPE", "sft_loss")
    monkeypatch.setattr(reporting, "_enqueue_timing", lambda *_a, **_k: None)

    recorder = RoleRecorder("driver", 0)
    for name in ("weight_sync", "reward_post_process", "train_models"):
        with recorder.phase(name):
            pass

    assert set(recorder.phases) == {"train_models"}


def test_preloop_recorders_close_before_queue_drains(monkeypatch):
    _configure(monkeypatch)
    timing_recorder._PRELOOP_RECORDERS.clear()
    queue = Queue()
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", queue)
    monkeypatch.setattr(reporting, "_REPORTER_STARTED", False)
    monkeypatch.setattr(reporting, "_REPORTER_THREAD", None)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_SENTINEL_QUEUED", False)
    monkeypatch.setattr(reporting, "_ensure_worker", lambda **_kwargs: None)

    with timing_recorder.recording_lane("driver", None) as recorder:
        with recorder.phase("initial_weight_sync"):
            pass

    queue.put(
        {
            "_url": "https://dashboard.test/api/timing-events",
            "training_run_id": "run-1",
            "storage_key": "pre-loop__driver",
            "final": False,
        }
    )
    reporting._drain_report_queue()

    queued = list(queue.queue)
    assert queued
    assert queued[0]["storage_key"] == "pre-loop__driver"
    assert queued[0]["final"] is True
