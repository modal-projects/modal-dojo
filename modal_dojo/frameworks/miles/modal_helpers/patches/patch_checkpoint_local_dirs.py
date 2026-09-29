"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/miles/backends/training_utils/checkpoint_io.py::write_checkpoint_dir
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/backends/training_utils/checkpoint_io.py")
MARKER = "PATCHED_TRAINING_GYM_CHECKPOINT_LOCAL_DIRS"
ANCHOR = "    try:\n        write_shards(checkpoint_dir)\n"
REPLACEMENT = f"""    try:
        # {MARKER}
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_shards(checkpoint_dir)
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one checkpoint write anchor")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: create checkpoint directory in every writer mount")


if __name__ == "__main__":
    apply()
