"""Completion records for native Beam adapter checkpoints on Modal Volumes."""

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import Lock

MARKER = "dojo_lora_complete.json"
LAYOUT = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
)


def _cursor_state(source) -> dict:
    state = {
        key: getattr(source, key)
        for key in (
            "sample_offset",
            "epoch_id",
            "sample_group_index",
            "sample_index",
            "metadata",
        )
    }
    if hasattr(source, "buffer"):
        state["buffer"] = source.buffer
    return deepcopy(state)


def remember_rollout_cursor(executor, rollout_id: int) -> None:
    """Remember the dataset position before the async producer consumes it."""
    if not executor.args.rollout_global_dataset:
        return
    if not hasattr(executor, "_dojo_cursor_lock"):
        executor._dojo_cursor_lock = Lock()
        executor._dojo_rollout_cursors = {}
    with executor._dojo_cursor_lock:
        executor._dojo_rollout_cursors[rollout_id] = _cursor_state(executor.data_source)
        for previous in list(executor._dojo_rollout_cursors):
            if previous < rollout_id - 1:
                del executor._dojo_rollout_cursors[previous]


def save_rollout_cursor(executor, rollout_id: int) -> None:
    """Save the next untrained batch's position, excluding in-flight prefetch."""
    if not executor.args.rollout_global_dataset:
        executor.data_source.save(rollout_id)
        return
    import torch
    from miles.rollout.data_source import compute_global_dataset_state_path

    with executor._dojo_cursor_lock:
        state = executor._dojo_rollout_cursors.get(rollout_id + 1)
        if state is None:
            state = _cursor_state(executor.data_source)
    path = Path(
        compute_global_dataset_state_path(executor.args.save, rollout_id=rollout_id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def complete_checkpoint(path: str) -> bool:
    root = Path(path)
    try:
        manifest = json.loads((root / MARKER).read_text())
        sizes = manifest["files"]
        world_size = manifest["world_size"]
        if not isinstance(world_size, int) or world_size < 1:
            return False
        required = {
            f"adapter/{prefix}{rank}.pt"
            for rank in range(world_size)
            for prefix in ("adapter_megatron_rank", "training_state_rank")
        }
        rollout_state = manifest.get("rollout_state")
        if rollout_state is not None:
            state_path = (
                root.parent
                / "rollout"
                / f"global_dataset_state_dict_{manifest['iteration']}.pt"
            )
            if state_path.stat().st_size != rollout_state:
                return False
        return required <= sizes.keys() and all(
            Path(name).parts[0] == "adapter"
            and ".." not in Path(name).parts
            and size > 0
            and (root / name).stat().st_size == size
            for name, size in sizes.items()
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def publish_checkpoint(args, rollout_id: int) -> None:
    """Called after native rank saves and rollout-state saves have finished."""
    if not args.save or not getattr(args, "lora_rank", None):
        return
    from modal import Volume

    volume_name = os.environ.get("MODAL_DOJO_CHECKPOINTS_VOLUME_NAME")
    volume = Volume.from_name(volume_name) if volume_name else None
    if volume:
        volume.commit()
        volume.reload()
    root = Path(args.save) / f"iter_{rollout_id:07d}"
    manifest = json.loads((root / "adapter" / "META.json").read_text())
    files = {}
    for rank in range(manifest["world_size"]):
        for prefix in ("adapter_megatron_rank", "training_state_rank"):
            name = f"adapter/{prefix}{rank}.pt"
            size = (root / name).stat().st_size
            if size <= 0:
                raise RuntimeError(f"Empty checkpoint shard: {root / name}")
            files[name] = size
    manifest["files"] = files
    if getattr(args, "rollout_global_dataset", False):
        state_path = (
            root.parent / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        )
        manifest["rollout_state"] = state_path.stat().st_size
    marker = root / MARKER
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest))
    temporary.replace(marker)
    tracker = root.parent / "latest_checkpointed_iteration.txt"
    tracker_tmp = tracker.with_suffix(".tmp")
    tracker_tmp.write_text(str(rollout_id))
    tracker_tmp.replace(tracker)
    if volume:
        volume.commit()


def validate_resume(args) -> None:
    adapter = Path(args.lora_adapter_path)
    if adapter.name != "adapter" or not complete_checkpoint(str(adapter.parent)):
        raise ValueError(f"Incomplete Beam resume checkpoint: {adapter.parent}")
    manifest = json.loads((adapter.parent / MARKER).read_text())
    if manifest["layout"] != {key: getattr(args, key) for key in LAYOUT}:
        raise ValueError("Native Beam resume requires the saved parallel layout")
    if args.no_load_optim or args.no_save_optim:
        raise ValueError("Beam resume requires saved optimizer and scheduler state")
