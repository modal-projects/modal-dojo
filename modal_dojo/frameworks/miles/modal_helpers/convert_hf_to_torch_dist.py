"""Thin wrapper around miles' convert_hf_to_torch_dist.py for Modal volumes.

The upstream script's release publication (inline shutil.move or the newer
publish_as_release helper) only sees local shards. The rename propagates a
deletion of iter_0000001/ that wipes other nodes' committed shards.

Setting SKIP_RELEASE_RENAME=1 suppresses the rename so all nodes commit
to iter_0000001/ additively. Megatron loads from iter_0000001/ via the
tracker file just fine.

TODO: replace the source rewrites with a native upstream option to retain the
iteration directory when one is available. Miles' publish_as_release still
renames it; its existing-directory fix does not synchronize Modal Volumes:
https://github.com/radixark/miles/blob/0cc65167f991ab1f0d228f6c9b648d89bc60a203/miles/backends/megatron_utils/checkpoint.py#L190-L199

Pipeline-parallel auto-inflation is suppressed upstream via CONVERT_KEEP_PP1,
which the launcher sets when it pins PP/TP explicitly (slime has no such env
var, hence the source patch in its wrapper).

CONVERT_DEQUANT_HF_WEIGHTS=1 dequantizes DeepSeek block-scaled fp8/fp4 weights
as mbridge reads them (see hf_block_dequant), for checkpoints that ship without
a bf16 export. CONVERT_DEQUANT_MXFP4=1 does the same for MXFP4
compressed-tensors experts (see hf_mxfp4_dequant), which Kimi-K3 ships.

CONVERT_LOCAL_STAGING=1 writes single-node conversions to local disk, then
copies shards to the Volume before publishing metadata and the tracker.

Without these options this wrapper is a transparent pass-through, apart
from local rank 0 logging host/cgroup memory so a SIGKILLed rank can be told
apart from a Python failure.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
from pathlib import Path

_UPSTREAM = "/root/miles/tools/convert_hf_to_torch_dist.py"
_MEM_LOG_INTERVAL_S = 30


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "n/a"


def _meminfo(field: str) -> str:
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith(field + ":"):
            return line.split(":", 1)[1].strip()
    return "n/a"


def _python_rss_gib() -> float:
    total_kb = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        status = _read(f"/proc/{pid}/status")
        if "Name:\tpython" not in status and "Name:\tpt_main_thread" not in status:
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total_kb += int(line.split()[1])
    return total_kb / 1024**2


def _log_memory() -> None:
    print(
        f"[convert-mem] cgroup max={_read('/sys/fs/cgroup/memory.max')}"
        f" current={_read('/sys/fs/cgroup/memory.current')}"
        f" MemTotal={_meminfo('MemTotal')} MemAvailable={_meminfo('MemAvailable')}"
        f" python_rss={_python_rss_gib():.1f}GiB",
        flush=True,
    )


def _start_memory_monitor() -> None:
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return

    def loop() -> None:
        while True:
            _log_memory()
            time.sleep(_MEM_LOG_INTERVAL_S)

    threading.Thread(target=loop, daemon=True, name="convert-mem").start()


def _load_upstream_source() -> str:
    with open(_UPSTREAM) as f:
        return f.read()


def keep_iteration_directory(src: str) -> str:
    """Keep all nodes' shards in one directory across both Miles converter APIs."""
    rename_calls = (
        "publish_as_release(args.save)",
        "shutil.move(source_dir, target_dir)",
    )
    matches = [call for call in rename_calls if src.count(call) == 1]
    tracker = 'f.write("release")'
    if len(matches) != 1 or src.count(tracker) != 1:
        raise RuntimeError(
            "Unknown Miles conversion publication code; refusing a multi-node rename"
        )
    return src.replace(matches[0], "pass  # SKIP_RELEASE_RENAME", 1).replace(
        tracker, 'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001', 1
    )


def _run_conversion(src: str) -> None:
    """Optionally stage a large, single-node conversion on local disk.

    All local ranks write the same temporary directory. Upstream's final
    barrier completes the save before rank zero copies it onto the Volume.
    Publish metadata and the iteration tracker only after every shard copies.
    """
    destination = None
    staging = None
    save_index = None
    if os.environ.get("CONVERT_LOCAL_STAGING") == "1":
        if int(os.environ.get("WORLD_SIZE", "1")) != int(
            os.environ.get("LOCAL_WORLD_SIZE", "1")
        ):
            raise ValueError("CONVERT_LOCAL_STAGING requires a single-node conversion")
        save_index = sys.argv.index("--save") + 1
        if not sys.argv[save_index].strip():
            raise ValueError("CONVERT_LOCAL_STAGING requires a nonempty --save path")
        destination = Path(sys.argv[save_index])
        key = hashlib.sha256(str(destination).encode()).hexdigest()[:16]
        staging = Path("/tmp/dojo-conversion") / key
        sys.argv[save_index] = str(staging)
    try:
        exec(
            compile(src, _UPSTREAM, "exec", dont_inherit=True),
            {"__name__": "__main__", "__file__": _UPSTREAM},
        )
        if destination is not None and os.environ.get("RANK", "0") == "0":
            print(
                f"Copying converted checkpoint from {staging} to {destination}",
                flush=True,
            )
            shutil.copytree(
                staging,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".metadata", "latest_checkpointed_iteration.txt"
                ),
            )
            for metadata in staging.rglob(".metadata"):
                shutil.copy2(metadata, destination / metadata.relative_to(staging))
            tracker = "latest_checkpointed_iteration.txt"
            shutil.copy2(staging / tracker, destination / tracker)
            shutil.rmtree(staging)
    finally:
        if save_index is not None:
            sys.argv[save_index] = str(destination)


def main() -> None:
    _start_memory_monitor()
    src = _load_upstream_source()
    if os.environ.get("SKIP_RELEASE_RENAME"):
        src = keep_iteration_directory(src)
    if os.environ.get("CONVERT_DEQUANT_HF_WEIGHTS"):
        # torchrun runs this file as a script, so its directory heads sys.path
        # while the package itself may not be importable under PYTHONPATH.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hf_block_dequant

        hf_block_dequant.install()
    if os.environ.get("CONVERT_DEQUANT_MXFP4"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hf_mxfp4_dequant

        hf_mxfp4_dequant.install()
    _run_conversion(src)


if __name__ == "__main__":
    main()
