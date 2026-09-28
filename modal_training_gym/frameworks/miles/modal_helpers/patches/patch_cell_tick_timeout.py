"""Extend Miles' cell tick timeout for the initial weights backup.

Kimi-K3's first release_memory_occupation copies frozen weights to host RAM
and exceeds the hardcoded 120-second timeout on eight-GPU nodes.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

MARKER = "PATCHED_CELL_TICK_TIMEOUT"
TIMEOUT = 1800.0

TARGET = pathlib.Path("/root/miles/miles/ray/rollout/inference_controller.py")
CONSTANT_RE = re.compile(r"^CELL_TICK_TIMEOUT_SECONDS = [\d.]+$", re.MULTILINE)
NEW_CONSTANT = f"CELL_TICK_TIMEOUT_SECONDS = {TIMEOUT}  # {MARKER}"

if not TARGET.exists():
    print(f"{TARGET} not found; skipping cell tick timeout patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("Cell tick timeout patch already applied")
    raise SystemExit(0)

src, replacements = CONSTANT_RE.subn(NEW_CONSTANT, src)
if not replacements:
    raise SystemExit(
        "Cell tick timeout patch did not match; miles' inference_controller.py "
        "has changed. Re-check CELL_TICK_TIMEOUT_SECONDS before shipping."
    )

TARGET.write_text(src)
print(f"Patched CELL_TICK_TIMEOUT_SECONDS -> {TIMEOUT}s")
