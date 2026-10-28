"""
image: radixark/miles:v0.1.2
commit: https://github.com/radixark/miles/commit/819cfd23ab74de9975931597894500632dd708fc
file: miles/ray/rollout/inference_controller.py, miles/ray/train/group.py

Allow cold Beam weight loading to outlast the hard-coded one-hour ready gate.
TODO: Remove when pinned Miles exposes controller readiness timeouts in config.
"""

from pathlib import Path

MARKER = "# MODAL_DOJO_BEAM_STARTUP"


def patch_source(source: str, target: str) -> str:
    if MARKER in source:
        return source
    constant = {
        "inference_controller.py": "CELLS_READY_TIMEOUT_SECONDS",
        "group.py": "_CELLS_READY_TIMEOUT_SECONDS",
    }[target]
    anchor = f"{constant} = 3600.0"
    if source.count(anchor) != 1:
        raise RuntimeError(f"Pinned Miles readiness timeout changed: {target}")
    replacement = f"""{MARKER}
import math as _dojo_math
import os as _dojo_os
{constant} = float(_dojo_os.environ.get("MILES_CELL_READY_TIMEOUT_SECONDS", "3600"))
if not _dojo_math.isfinite({constant}) or {constant} <= 0:
    raise ValueError("MILES_CELL_READY_TIMEOUT_SECONDS must be positive and finite")"""
    return source.replace(anchor, replacement)


def main() -> None:
    for relative in ("rollout/inference_controller.py", "train/group.py"):
        path = Path("/root/miles/miles/ray") / relative
        path.write_text(patch_source(path.read_text(), path.name))
        print(f"Applied configurable Beam readiness timeout to {path}")


if __name__ == "__main__":
    main()
