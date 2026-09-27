"""Patch miles' inference-controller tick timeout for engines that release slowly.

``miles/ray/rollout/inference_controller.py`` bounds every cell tick with a
hardcoded ``CELL_TICK_TIMEOUT_SECONDS = 120.0`` and no CLI flag. The first tick
after an engine comes up is ``release_memory_occupation``: with
``--enable-memory-saver`` and the weights CPU backup miles turns on for
colocation, a TP16 Kimi-K3 engine has ~95 GB of weights a rank to copy to host,
eight ranks a node on Modal B300 (upstream validated four). That takes longer
than 120 s, the controller logs ``Ticking cell ... failed`` on a
``TimeoutError``, keeps re-ticking against an engine still mid-release, and a
later probe request lands on released memory (``Pointer argument cannot be
accessed from Triton (cpu tensor?)``). Raising the bound is safe: a tick that
finishes returns immediately, and a dead engine still fails on its own.

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
