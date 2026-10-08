import pytest

from modal_dojo.common import torch_dist_checkpoint as tdc


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


def test_commit_with_retries_returns_none_on_first_success() -> None:
    calls: list[str] = []
    clock = FakeClock()
    error = tdc.commit_volume_with_retries(
        "vol",
        commit=calls.append,
        timeout_seconds=60,
        clock=clock,
        sleep=clock.sleep,
    )
    assert error is None
    assert calls == ["vol"]


def test_commit_with_retries_retries_until_success() -> None:
    attempts: list[int] = []

    def flaky(_: str) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("snapshot file upload timeout")

    clock = FakeClock()
    error = tdc.commit_volume_with_retries(
        "vol",
        commit=flaky,
        timeout_seconds=3600,
        retry_delay_seconds=30,
        clock=clock,
        sleep=clock.sleep,
    )
    assert error is None
    assert len(attempts) == 3
    assert clock.now == 1000.0 + 2 * 30


def test_commit_with_retries_gives_up_at_deadline() -> None:
    attempts: list[int] = []

    def failing(_: str) -> None:
        attempts.append(1)
        raise TimeoutError("snapshot file upload timeout")

    clock = FakeClock()
    error = tdc.commit_volume_with_retries(
        "vol",
        commit=failing,
        timeout_seconds=100,
        retry_delay_seconds=30,
        clock=clock,
        sleep=clock.sleep,
    )
    assert error == "TimeoutError: snapshot file upload timeout"
    # Attempts at t=0, 30, 60, 90 and a last one at the t=100 deadline; the
    # final sleep is clamped so the deadline is never overshot.
    assert len(attempts) == 5
    assert clock.now == 1100.0


def test_commit_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tdc.COMMIT_TIMEOUT_ENV, raising=False)
    assert tdc.commit_timeout_seconds() == tdc.DEFAULT_COMMIT_TIMEOUT_SECONDS
    monkeypatch.setenv(tdc.COMMIT_TIMEOUT_ENV, "90")
    assert tdc.commit_timeout_seconds() == 90.0


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
    tdc.commit_checkpoint_volume_across_ranks(
        "vol", dist, lambda: None, commit=commits.append
    )
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
    dist = FakeDist(
        is_coordinator=True,
        remote_errors=["TimeoutError: snapshot file upload timeout"],
    )
    commits: list[str] = []
    with pytest.raises(RuntimeError, match="shard commit failed"):
        tdc.commit_checkpoint_volume_across_ranks(
            "vol", dist, lambda: None, commit=commits.append
        )
    assert commits == []


def test_shard_failure_after_retries_is_raised_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    dist = FakeDist(is_coordinator=False)
    with pytest.raises(RuntimeError, match="shard commit failed.*upload timeout"):
        tdc.commit_checkpoint_volume_across_ranks(
            "vol",
            dist,
            lambda: None,
            commit=lambda _: "TimeoutError: snapshot file upload timeout",
        )
