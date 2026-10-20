"""Expose pinned TMS disk backups for colocated native-LoRA serving weights.

commit: https://github.com/sgl-project/sglang/commit/34033fe90b90f1bb64e63337268beada57f21f84
file: python/sglang/srt/utils/torch_memory_saver_adapter.py

The trainer and serving CPU backups together exceeded 1792 GiB/node. Keep
serving backups on local disk while the trainer uses CPU memory.
TODO: Remove when SGLang exposes TMS disk-backed weight regions directly.
"""

from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/utils/torch_memory_saver_adapter.py"
)
MARKER = "PATCHED_MODAL_DOJO_SGLANG_WEIGHTS_DISK_BACKUP"
ANCHOR = (
    "        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)"
)
REPLACEMENT = f"""        # {MARKER}
        import os

        disk_root = os.environ.get("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR")
        if tag == "weights" and enable_cpu_backup and disk_root:
            # TMS directories are process-global. Separate each serving worker
            # from the trainer and other TP ranks on the same node.
            disk_dir = os.path.join(disk_root, str(os.getpid()))
            os.makedirs(disk_dir, exist_ok=True)
            _memory_saver.set_disk_backup_dir(disk_dir)
            return _memory_saver.region(
                tag=tag, enable_cpu_backup=False, enable_disk_backup=True
            )
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected pinned SGLang memory-saver hook")
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
