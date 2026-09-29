"""Thin wrapper around miles' convert_hf_to_torch_dist.py for Modal volumes.

The upstream script's shutil.move(iter_0000001 -> release) only sees local shards,
which poisons the volume state for multi-node conversions (the rename propagates a
deletion of iter_0000001/ that wipes other nodes' committed shards).

Setting SKIP_RELEASE_RENAME=1 suppresses the rename so all nodes commit
to iter_0000001/ additively. Megatron loads from iter_0000001/ via the
tracker file just fine.

Pipeline-parallel auto-inflation is suppressed upstream via CONVERT_KEEP_PP1,
which the launcher sets when it pins PP/TP explicitly (slime has no such env
var, hence the source patch in its wrapper).

CONVERT_DEQUANT_HF_WEIGHTS=1 dequantizes DeepSeek block-scaled fp8/fp4 weights
as mbridge reads them (see hf_block_dequant), for checkpoints that ship without
a bf16 export.

When neither variable is set this wrapper is a transparent pass-through, apart
from local rank 0 logging host/cgroup memory so a SIGKILLed rank can be told
apart from a Python failure.
"""

from __future__ import annotations

import os
import sys
import threading
import time

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


def main() -> None:
    _start_memory_monitor()
    src = _load_upstream_source()
    if os.environ.get("SKIP_RELEASE_RENAME"):
        src = src.replace(
            "shutil.move(source_dir, target_dir)",
            "pass  # SKIP_RELEASE_RENAME",
        )
        src = src.replace(
            'f.write("release")',
            'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001',
        )
    if os.environ.get("CONVERT_DEQUANT_HF_WEIGHTS"):
        # torchrun runs this file as a script, so its directory heads sys.path
        # while the package itself may not be importable under PYTHONPATH.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hf_block_dequant

        hf_block_dequant.install()
    exec(
        compile(src, _UPSTREAM, "exec"), {"__name__": "__main__", "__file__": _UPSTREAM}
    )


if __name__ == "__main__":
    main()
