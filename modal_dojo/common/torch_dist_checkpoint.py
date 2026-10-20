from __future__ import annotations

import os
import pickle
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


def commit_checkpoint_volume_across_ranks(
    volume_name: str | None,
    dist_wrapper: Any,
    barrier: Callable[[], None],
) -> None:
    if not volume_name:
        return

    shard_commit_error = None
    if _is_node_leader() and not dist_wrapper.is_coordinator:
        try:
            _commit_volume(volume_name)
        except Exception as exc:
            shard_commit_error = _error_text(exc)
    _raise_gathered_errors(
        dist_wrapper,
        shard_commit_error,
        "Checkpoint Volume shard commit failed",
    )
    barrier()

    coordinator_commit_error = None
    if _is_node_leader() and dist_wrapper.is_coordinator:
        try:
            _commit_volume(volume_name)
        except Exception as exc:
            coordinator_commit_error = _error_text(exc)
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
    timeout = os.environ.get("MODAL_DOJO_CHECKPOINT_COMMIT_TIMEOUT_SECONDS")
    if timeout is not None and dist.is_initialized():
        # Terabyte-scale Volume publication can outlast the GPU collective
        # timeout. Keep this storage wait off NCCL and bound it independently.
        seconds = int(timeout)
        if seconds <= 0:
            raise ValueError("Checkpoint commit timeout must be positive")
        group = dist.new_group(backend="gloo", timeout=timedelta(seconds=seconds))
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
        )
    finally:
        if group is not None:
            dist.destroy_process_group(group)
