"""Create checkpoint directories in every writer's Modal Volume mount.

Rank zero's directory creation is not immediately visible on other nodes.
Create it locally after rank-zero preparation, inside write-error collection.
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
