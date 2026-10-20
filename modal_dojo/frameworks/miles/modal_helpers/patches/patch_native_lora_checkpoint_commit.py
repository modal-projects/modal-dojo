"""Commit native LoRA shards from every Modal node before save returns.

Miles 65e785ff5e794a7081383e18e70c4555cb1fe0fb saves adapters directly,
bypassing Megatron's save_checkpoint and its existing Volume commit wrapper.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/backends/megatron_utils/lora/utils.py")
MARKER = "PATCHED_MODAL_DOJO_NATIVE_LORA_CHECKPOINT_COMMIT"
ANCHOR = "    write_checkpoint_dir(save_dir, write_shards)\n"
REPLACEMENT = (
    ANCHOR
    + f"""    # {MARKER}
    from modal_dojo.common.torch_dist_checkpoint import commit_latest_checkpoint_volume

    commit_latest_checkpoint_volume()
"""
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one native LoRA checkpoint write")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
