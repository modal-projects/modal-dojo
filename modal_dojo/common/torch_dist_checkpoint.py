from __future__ import annotations

import os
from collections.abc import Collection
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
    @property
    def is_coordinator(self) -> bool:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0

    def all_gather_object(self, error: str | None) -> list[str | None]:
        import torch.distributed as dist

        if not dist.is_initialized():
            return [error]
        gathered: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, error)
        return gathered


def commit_latest_checkpoint_volume() -> None:
    import torch.distributed as dist

    volume_name = os.environ.get("TRAINING_GYM_CHECKPOINTS_VOLUME_NAME")
    barrier = dist.barrier if dist.is_initialized() else lambda: None
    commit_checkpoint_volume_across_ranks(
        volume_name,
        _TorchDistributed(),
        barrier,
    )
