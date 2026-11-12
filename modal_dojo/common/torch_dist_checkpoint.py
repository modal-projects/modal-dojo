from __future__ import annotations

import os
import pickle
import time
from collections.abc import Collection
from datetime import timedelta
from functools import partial
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


def is_complete_torch_dist_checkpoint_storage(
    checkpoint_dir: str | os.PathLike[str],
) -> bool:
    """Check every referenced storage range in a trusted torch_dist checkpoint.

    Metadata can become visible before another node commits its shards to a
    shared Volume. Checking filenames alone cannot establish completeness in
    that case. This reads metadata (a pickle), but never reads tensor payloads.
    Only use it on checkpoints produced by the trusted conversion pipeline.
    """
    root = Path(checkpoint_dir).resolve()
    if not is_complete_torch_dist_checkpoint_dir(root):
        return False
    try:
        with (root / ".metadata").open("rb") as handle:
            metadata = pickle.load(handle)
        storage = metadata.storage_data
        if not isinstance(storage, dict) or not storage:
            return False
        required_sizes: dict[Path, int] = {}
        for location in storage.values():
            relative_path = Path(location.relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return False
            offset, length = location.offset, location.length
            if not isinstance(offset, int) or not isinstance(length, int):
                return False
            if offset < 0 or length < 0:
                return False
            required_sizes[relative_path] = max(
                required_sizes.get(relative_path, 0), offset + length
            )
        # A shard contains thousands of ranges. Resolve and stat each file once
        # instead of repeating remote filesystem lookups for every tensor.
        for relative_path, size in required_sizes.items():
            shard = (root / relative_path).resolve()
            if not shard.is_relative_to(root) or shard.suffix != ".distcp":
                return False
            if not shard.is_file() or shard.stat().st_size < size:
                return False
        return True
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        TypeError,
        ValueError,
    ):
        return False


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


COMMIT_TIMEOUT_ENV = "MODAL_DOJO_CHECKPOINT_COMMIT_TIMEOUT_SECONDS"
COMMIT_RETRY_DELAY_SECONDS = 30.0
# Ranks waiting on the gloo group outlive the retry deadline by this much, so
# the final retry's sleep and the error gather finish before the group times out.
COMMIT_GROUP_SLACK_SECONDS = 10 * 60


def _commit_volume_once(volume_name: str) -> str | None:
    try:
        _commit_volume(volume_name)
    except Exception as exc:
        return _error_text(exc)
    return None


def commit_volume_with_retries(
    volume_name: str,
    timeout_seconds: float,
    *,
    commit: Callable[[str], None] = _commit_volume,
    retry_delay_seconds: float = COMMIT_RETRY_DELAY_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Commit ``volume_name`` until it succeeds or ``timeout_seconds`` runs out.

    A commit cut short by a per-file upload timeout (blob-store throttling on a
    multi-GB shard) is safe to repeat: files already published are skipped, so
    a retry only uploads what is still missing. A retry is only started when
    the time left covers the delay and another attempt as long as the last
    one, so the slowest node finishes inside the gloo group's timeout.

    Returns ``None`` on success, otherwise the text of the last error."""
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
            if attempt > 1:
                print(
                    f"Checkpoint Volume commit succeeded on attempt {attempt}",
                    flush=True,
                )
            return None
        last_attempt = clock() - started
        remaining = deadline - clock()
        if remaining < retry_delay_seconds + last_attempt:
            print(
                f"Checkpoint Volume commit gave up after {attempt} attempt(s): {error}",
                flush=True,
            )
            return error
        print(
            f"Checkpoint Volume commit attempt {attempt} failed, retrying in "
            f"{retry_delay_seconds:.0f}s ({remaining:.0f}s left): {error}",
            flush=True,
        )
        sleep(retry_delay_seconds)


def commit_checkpoint_volume_across_ranks(
    volume_name: str | None,
    dist_wrapper: Any,
    barrier: Callable[[], None],
    *,
    commit: Callable[[str], str | None] = _commit_volume_once,
) -> None:
    """Commit the checkpoint Volume from every node, shards before metadata.

    ``commit`` returns the error text, or ``None`` on success. The coordinator
    commits last because its commit publishes ``.metadata``, the completeness
    marker, and it skips that commit when any shard commit failed."""
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
    def __init__(self, group: Any = None) -> None:
        self.group = group

    @property
    def is_coordinator(self) -> bool:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0

    def all_gather_object(self, error: str | None) -> list[str | None]:
        import torch.distributed as dist

        if not dist.is_initialized():
            return [error]
        gathered: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, error, group=self.group)
        return gathered


def commit_latest_checkpoint_volume() -> None:
    volume_name = os.environ.get("MODAL_DOJO_CHECKPOINTS_VOLUME_NAME")
    if not volume_name:
        return

    import torch.distributed as dist

    group = None
    commit: Callable[[str], str | None] = _commit_volume_once
    timeout = os.environ.get(COMMIT_TIMEOUT_ENV)
    if timeout is not None:
        # Terabyte-scale Volume publication can outlast the GPU collective
        # timeout. Keep this storage wait off NCCL and bound it independently,
        # retrying failed commits until the timeout runs out. Without the
        # timeout, ranks wait on the NCCL group, whose watchdog would kill the
        # job long before retries could help, so commit once.
        seconds = int(timeout)
        if seconds <= 0:
            raise ValueError("Checkpoint commit timeout must be positive")
        commit = partial(commit_volume_with_retries, timeout_seconds=seconds)
        if dist.is_initialized():
            group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=seconds + COMMIT_GROUP_SLACK_SECONDS),
            )
    try:
        barrier = (
            (lambda: dist.barrier(group=group))
            if dist.is_initialized()
            else lambda: None
        )
        commit_checkpoint_volume_across_ranks(
            volume_name,
            _TorchDistributed(group),
            barrier,
            commit=commit,
        )
    finally:
        if group is not None:
            dist.destroy_process_group(group)
