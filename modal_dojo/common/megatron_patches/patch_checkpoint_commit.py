from pathlib import Path

_CHECKPOINTING = Path("/root/Megatron-LM/megatron/training/checkpointing.py")
_ASYNC_UTILS = Path("/root/Megatron-LM/megatron/training/async_utils.py")
_SAVE_MARKER = "PATCHED_TRAINING_GYM_CHECKPOINT_COMMIT"
_FINALIZE_MARKER = "PATCHED_TRAINING_GYM_ASYNC_FINALIZE_COMMIT"
_SAVE_WRAP = """

# PATCHED_TRAINING_GYM_CHECKPOINT_COMMIT
def save_checkpoint(*args, **kwargs):
    from megatron.training import get_args
    from modal_dojo.common.torch_dist_checkpoint import (
        commit_latest_checkpoint_volume,
    )

    result = _training_gym_save_checkpoint(*args, **kwargs)
    if not getattr(get_args(), "async_save", False):
        commit_latest_checkpoint_volume()
    return result
"""
_FINALIZE_WRAP = """

# PATCHED_TRAINING_GYM_ASYNC_FINALIZE_COMMIT
def maybe_finalize_async_save(*args, **kwargs):
    from modal_dojo.common.torch_dist_checkpoint import (
        commit_latest_checkpoint_volume,
    )

    had_pending = not is_empty_async_queue()
    result = _training_gym_maybe_finalize_async_save(*args, **kwargs)
    if had_pending:
        commit_latest_checkpoint_volume()
    return result
"""


def _wrap(path: Path, original: str, renamed: str, marker: str, wrap: str) -> None:
    if not path.is_file():
        print(f"WARNING: {path} is absent; skipping {marker}")
        return
    source = path.read_text()
    if marker in source:
        print(f"{path.name} already has {marker}")
        return
    if f"def {original}(" not in source:
        raise RuntimeError(f"Megatron {original} not found in {path}")
    source = source.replace(f"def {original}(", f"def {renamed}(", 1)
    source += wrap
    compile(source, str(path), "exec")
    path.write_text(source)
    print(f"Patched {path} ({original})")


def main() -> None:
    _wrap(
        _CHECKPOINTING,
        "save_checkpoint",
        "_training_gym_save_checkpoint",
        _SAVE_MARKER,
        _SAVE_WRAP,
    )
    _wrap(
        _ASYNC_UTILS,
        "maybe_finalize_async_save",
        "_training_gym_maybe_finalize_async_save",
        _FINALIZE_MARKER,
        _FINALIZE_WRAP,
    )


if __name__ == "__main__":
    main()
