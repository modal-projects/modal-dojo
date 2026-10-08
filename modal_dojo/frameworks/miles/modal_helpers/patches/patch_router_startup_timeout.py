"""Patch Miles rollout router startup to wait longer for its port.

``miles/ray/rollout/router_manager.py`` hardcodes the readiness bound with no
CLI flag: older images inline ``wait_for_server_ready(..., timeout=30)`` at
both call sites, newer ones (radixark/miles#2604 onward) route both through a
``_SERVER_READY_TIMEOUT_SECS = 120`` module constant.
The router is spawned while the Megatron actors load the checkpoint, so on a
large model the child is alive but starved and has not bound its port yet.
Raising the bound is safe: the wait returns as soon as the port accepts, and
still fails fast if the child actually dies.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

MARKER = "PATCHED_ROUTER_STARTUP_TIMEOUT"
TIMEOUT = 600

TARGET = pathlib.Path("/root/miles/miles/ray/rollout/router_manager.py")

OLD = "wait_for_server_ready(router_ip, router_port, process, timeout=30)"
NEW = (
    f"wait_for_server_ready(router_ip, router_port, process, timeout={TIMEOUT})"
    f"  # {MARKER}"
)
OLD_SESSION = "wait_for_server_ready(ip, port, process, timeout=30)"
NEW_SESSION = f"wait_for_server_ready(ip, port, process, timeout={TIMEOUT})  # {MARKER}"
CONSTANT_RE = re.compile(r"^_SERVER_READY_TIMEOUT_SECS = \d+$", re.MULTILINE)
NEW_CONSTANT = f"_SERVER_READY_TIMEOUT_SECS = {TIMEOUT}  # {MARKER}"

if not TARGET.exists():
    print(f"{TARGET} not found; skipping router startup timeout patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("Router startup timeout patch already applied")
    raise SystemExit(0)

replacements = 0
for old, new in ((OLD, NEW), (OLD_SESSION, NEW_SESSION)):
    if old in src:
        src = src.replace(old, new)
        replacements += 1
src, constant_replacements = CONSTANT_RE.subn(NEW_CONSTANT, src)
replacements += constant_replacements

if not replacements:
    raise SystemExit(
        "Router startup timeout patch did not match; miles' router_manager.py "
        "has changed. Re-check the server readiness timeout before shipping."
    )

TARGET.write_text(src)
print(f"Patched {replacements} router readiness timeout site(s) -> {TIMEOUT}s")
