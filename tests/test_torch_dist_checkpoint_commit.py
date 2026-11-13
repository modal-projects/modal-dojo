import sys
import types

import pytest

from modal_dojo.common import torch_dist_checkpoint as tdc

UPLOAD_TIMEOUT = "TimeoutError: snapshot file upload timeout"


class FakeDist:
    def __init__(self, *, is_coordinator: bool, remote_errors=()) -> None:
        self.is_coordinator = is_coordinator
        self.remote_errors = list(remote_errors)
        self.gathers: list[str | None] = []

    def all_gather_object(self, error):
        self.gathers.append(error)
        return [error, *self.remote_errors]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _retry(commit, clock, timeout_seconds):
    return tdc.commit_volume_with_retries(
        "vol",
        timeout_seconds,
        commit=commit,
        retry_delay_seconds=30,
        clock=clock,
        sleep=clock.sleep,
    )


def test_retries_return_none_on_first_success() -> None:
    calls: list[str] = []
    assert _retry(calls.append, FakeClock(), 60) is None
    assert calls == ["vol"]


def test_retries_until_success() -> None:
    attempts: list[int] = []

    def flaky(_: str) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("snapshot file upload timeout")

    clock = FakeClock()
    assert _retry(flaky, clock, 3600) is None
    assert len(attempts) == 3
    assert clock.now == 1000.0 + 2 * 30


def test_retries_give_up_before_deadline() -> None:
    attempts: list[int] = []

    def failing(_: str) -> None:
        attempts.append(1)
        raise TimeoutError("snapshot file upload timeout")

    clock = FakeClock()
    assert _retry(failing, clock, 100) == UPLOAD_TIMEOUT
    # Instant failures at t=0, 30, 60, 90. At t=90 only 10s remain, less than
    # the 30s delay, so no fifth attempt starts.
    assert len(attempts) == 4
    assert clock.now == 1090.0


def test_no_retry_when_last_attempt_would_not_fit() -> None:
    attempts: list[int] = []
    clock = FakeClock()

    def slow_failure(_: str) -> None:
        attempts.append(1)
        clock.now += 1800  # the worker's 30-minute per-file upload cap
        raise TimeoutError("snapshot file upload timeout")

    # After an 1800s failure, a 3600s timeout leaves 1800s: too little for the
    # 30s delay plus another 1800s attempt. A 4800s timeout leaves 3000s, so
    # one retry runs, and then 1170s is too little for a third.
    assert _retry(slow_failure, clock, 3600) == UPLOAD_TIMEOUT
    assert len(attempts) == 1

    attempts.clear()
    clock = FakeClock()
    assert _retry(slow_failure, clock, 4800) == UPLOAD_TIMEOUT
    assert len(attempts) == 2
    assert clock.now == 1000.0 + 3630


def test_across_ranks_noop_without_volume() -> None:
    dist = FakeDist(is_coordinator=True)
    barriers: list[int] = []
    tdc.commit_checkpoint_volume_across_ranks(
        None, dist, lambda: barriers.append(1), commit=lambda _: None
    )
    assert dist.gathers == []
    assert barriers == []


def test_shard_leader_commits_then_waits_for_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist = FakeDist(is_coordinator=False)
    commits: list[str] = []
    barriers: list[int] = []

    def commit(name: str) -> None:
        commits.append(name)

    tdc.commit_checkpoint_volume_across_ranks(
        "vol", dist, lambda: barriers.append(1), commit=commit
    )
    assert commits == ["vol"]
    assert dist.gathers == [None, None]
    assert barriers == [1, 1]


def test_non_leader_rank_never_commits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")
    dist = FakeDist(is_coordinator=False)
    commits: list[str] = []

    def commit(name: str) -> None:
        commits.append(name)

    tdc.commit_checkpoint_volume_across_ranks("vol", dist, lambda: None, commit=commit)
    assert commits == []
    assert dist.gathers == [None, None]


def test_coordinator_commits_only_after_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist = FakeDist(is_coordinator=True)
    order: list[str] = []

    def commit(name: str) -> None:
        order.append(f"commit:{name}")

    tdc.commit_checkpoint_volume_across_ranks(
        "vol", dist, lambda: order.append("barrier"), commit=commit
    )
    assert order == ["barrier", "commit:vol", "barrier"]


def test_coordinator_skips_metadata_commit_when_a_shard_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist = FakeDist(is_coordinator=True, remote_errors=[UPLOAD_TIMEOUT])
    commits: list[str] = []

    def commit(name: str) -> None:
        commits.append(name)

    with pytest.raises(RuntimeError, match="shard commit failed"):
        tdc.commit_checkpoint_volume_across_ranks(
            "vol", dist, lambda: None, commit=commit
        )
    assert commits == []


def test_shard_failure_is_raised_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist = FakeDist(is_coordinator=False)
    with pytest.raises(RuntimeError, match=r"shard commit failed.*upload timeout"):
        tdc.commit_checkpoint_volume_across_ranks(
            "vol", dist, lambda: None, commit=lambda _: UPLOAD_TIMEOUT
        )


def test_latest_commit_retries_only_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failing(name: str) -> None:
        calls.append(name)
        raise TimeoutError("snapshot file upload timeout")

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_initialized = lambda: False  # type: ignore[attr-defined]
    fake_torch = types.ModuleType("torch")
    fake_torch.distributed = fake_dist  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setattr(tdc, "_commit_volume", failing)
    monkeypatch.setenv("MODAL_DOJO_CHECKPOINTS_VOLUME_NAME", "vol")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    monkeypatch.delenv(tdc.COMMIT_TIMEOUT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="coordinator commit failed"):
        tdc.commit_latest_checkpoint_volume()
    assert calls == ["vol"]

    calls.clear()

    def fake_retries(name: str, timeout_seconds: float) -> str:
        calls.append(f"retry:{name}:{timeout_seconds}")
        return UPLOAD_TIMEOUT

    monkeypatch.setattr(tdc, "commit_volume_with_retries", fake_retries)
    monkeypatch.setenv(tdc.COMMIT_TIMEOUT_ENV, "3600")
    with pytest.raises(RuntimeError, match="coordinator commit failed"):
        tdc.commit_latest_checkpoint_volume()
    assert calls == ["retry:vol:3600"]
