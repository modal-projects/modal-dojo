"""
image: radixark/miles:dev-202610180955
commit: https://github.com/radixark/miles/commit/24bd5fb7166a185f309399e52f43b100e4369203
file: miles/backends/training_utils/checkpoint/io.py

Create native adapter directories in every node's Modal Volume view and
commit all rank shards. Native LoRA bypasses Megatron's checkpoint hook.
TODO: remove when Miles exposes a native post-save hook and creates each
node's destination before writing shards.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_NATIVE_LORA_CHECKPOINT"


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    anchor = "        write_shards(checkpoint_dir)"
    if source.count(anchor) != 1 or source.count("def write_checkpoint_dir(") != 1:
        raise ValueError("Miles native checkpoint writer changed")
    source = source.replace(
        anchor,
        "        checkpoint_dir.mkdir(parents=True, exist_ok=True)\n" + anchor,
    ).replace("def write_checkpoint_dir(", "def _dojo_write_checkpoint_dir(", 1)
    source += f"""

# {MARKER}
def write_checkpoint_dir(*args, **kwargs):
    from modal_dojo.common.torch_dist_checkpoint import commit_latest_checkpoint_volume

    result = _dojo_write_checkpoint_dir(*args, **kwargs)
    commit_latest_checkpoint_volume()
    return result
"""
    compile(source, "io.py", "exec")
    return source


def main() -> None:
    path = Path("/root/miles/miles/backends/training_utils/checkpoint/io.py")
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
