from __future__ import annotations

import os
import time
from collections.abc import Collection
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

TORCH_DIST_TRACKER_NAME = "latest_checkpointed_iteration.txt"


def is_complete_torch_dist_checkpoint(names: Collection[str]) -> bool:
    """``.metadata`` is written last, so it separates a finished save from a crashed
    one. ``common.pt`` is not required: newer megatron-core folds the common state
    into the torch_dist metadata and writes no such file."""
    return ".metadata" in names and any(name.endswith(".distcp") for name in names)


def is_complete_torch_dist_checkpoint_dir(
    checkpoint_dir: str | os.PathLike[str],
) -> bool:
    try:
        names = {entry.name for entry in Path(checkpoint_dir).iterdir()}
    except OSError:
        return False
    return is_complete_torch_dist_checkpoint(names)


def parse_torch_dist_iteration(name: str) -> int | None:
    if not name.startswith("iter_"):
        return None
    try:
        return int(name.removeprefix("iter_"))
    except ValueError:
        return None


def parse_torch_dist_tracker(text: str) -> int | None:
    text = text.strip()
    if text == "release" or not text.isdigit():
        return None
    return int(text)


def _is_node_leader() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank is None or int(local_rank) == 0


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _raise_gathered_errors(
    dist_wrapper: Any,
    error: str | None,
    message: str,
) -> None:
    errors = dist_wrapper.all_gather_object(error)
    if any(errors):
        raise RuntimeError(f"{message}: {errors}")


def _commit_volume(volume_name: str) -> None:
    from modal import Volume

    Volume.from_name(volume_name, create_if_missing=False).commit()


# A Volume commit uploads every dirty file to blob storage before it returns.
# Multi-GB checkpoint shards can take far longer than the NCCL collective
# timeout when the blob store throttles, and a commit interrupted by a
# per-file upload timeout is safe to retry: files already published are
# skipped, so each retry only re-uploads what is still missing.
COMMIT_TIMEOUT_ENV = "MODAL_DOJO_CHECKPOINT_COMMIT_TIMEOUT_SECONDS"
DEFAULT_COMMIT_TIMEOUT_SECONDS = 2 * 60 * 60
COMMIT_RETRY_DELAY_SECONDS = 30.0


def commit_timeout_seconds() -> float:
    raw = os.environ.get(COMMIT_TIMEOUT_ENV)
    return float(raw) if raw else float(DEFAULT_COMMIT_TIMEOUT_SECONDS)


def commit_volume_with_retries(
    volume_name: str,
    *,
    commit: Callable[[str], None] = _commit_volume,
    timeout_seconds: float | None = None,
    retry_delay_seconds: float = COMMIT_RETRY_DELAY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Commit ``volume_name`` until it succeeds or ``timeout_seconds`` elapses.

    Returns ``None`` on success, otherwise the text of the last error."""
    if timeout_seconds is None:
        timeout_seconds = commit_timeout_seconds()
    deadline = clock() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        started = clock()
        try:
            commit(volume_name)
        except Exception as exc:
            error = _error_text(exc)
        else:
            print(
                f"Checkpoint Volume commit succeeded after {attempt} attempt(s) "
                f"({clock() - started:.0f}s)",
                flush=True,
            )
            return None
        remaining = deadline - clock()
        if remaining <= 0:
            print(
                f"Checkpoint Volume commit gave up after {attempt} attempt(s): {error}",
                flush=True,
            )
            return error
        print(
            f"Checkpoint Volume commit attempt {attempt} failed, retrying "
            f"({remaining:.0f}s left): {error}",
            flush=True,
        )
        sleep(min(retry_delay_seconds, remaining))


def commit_checkpoint_volume_across_ranks(
    volume_name: str | None,
    dist_wrapper: Any,
    barrier: Callable[[], None],
    *,
    commit: Callable[[str], str | None] = commit_volume_with_retries,
) -> None:
    """Commit the checkpoint Volume from every node, shards before metadata.

    ``dist_wrapper`` and ``barrier`` must not run on the NCCL process group used
    for training: ranks wait here for as long as the slowest commit takes, so
    the group they wait on needs a timeout at least as long as the commit
    timeout (see :class:`_TorchDistributed`).

    Node leaders other than the coordinator commit first, so the coordinator's
    commit (which publishes ``.metadata``, the completeness marker) only happens
    once every shard is published; a checkpoint missing ``.metadata`` is never
    treated as loadable."""
    if not volume_name:
        return

    shard_commit_error = None
    if _is_node_leader() and not dist_wrapper.is_coordinator:
        shard_commit_error = commit(volume_name)
    _raise_gathered_errors(
        dist_wrapper,
        shard_commit_error,
        "Checkpoint Volume shard commit failed",
    )
    barrier()

    coordinator_commit_error = None
    if _is_node_leader() and dist_wrapper.is_coordinator:
        coordinator_commit_error = commit(volume_name)
    _raise_gathered_errors(
        dist_wrapper,
        coordinator_commit_error,
        "Checkpoint Volume coordinator commit failed",
    )
    barrier()


class _TorchDistributed:
    """Commit coordination on a dedicated gloo group.

    Every rank blocks in these collectives while node leaders commit, so they
    run on a CPU-side gloo group whose timeout covers the full commit timeout
    rather than on the default (NCCL) group, where the watchdog would kill the
    job after ``distributed_timeout_minutes``."""

    _group: Any = None

    def __init__(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            timeout_seconds = commit_timeout_seconds()
        self._timeout_seconds = timeout_seconds

    @property
    def is_coordinator(self) -> bool:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0

    def _commit_group(self) -> Any:
        import torch.distributed as dist

        if _TorchDistributed._group is None:
            # Slack past the commit deadline for the retry sleep and gather.
            timeout = timedelta(seconds=self._timeout_seconds + 10 * 60)
            _TorchDistributed._group = dist.new_group(backend="gloo", timeout=timeout)
        return _TorchDistributed._group

    def all_gather_object(self, error: str | None) -> list[str | None]:
        import torch.distributed as dist

        if not dist.is_initialized():
            return [error]
        gathered: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, error, group=self._commit_group())
        return gathered

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier(group=self._commit_group())


def commit_latest_checkpoint_volume() -> None:
    volume_name = os.environ.get("MODAL_DOJO_CHECKPOINTS_VOLUME_NAME")
    if not volume_name:
        return
    dist_wrapper = _TorchDistributed()
    commit_checkpoint_volume_across_ranks(
        volume_name,
        dist_wrapper,
        dist_wrapper.barrier,
    )
