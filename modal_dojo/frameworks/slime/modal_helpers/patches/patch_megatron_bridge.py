# Patch slime_plugins.megatron_bridge to skip a missing eager import.

from __future__ import annotations

from pathlib import Path

CANDIDATES = [
    Path("/root/slime_plugins/megatron_bridge/__init__.py"),
    Path("/root/slime/slime_plugins/megatron_bridge/__init__.py"),
    Path(
        "/usr/local/lib/python3.12/dist-packages/slime_plugins/megatron_bridge/__init__.py"
    ),
]

OLD = "import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401"
NEW = (
    "try:\n"
    "    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # PATCHED_GLM4V_BRIDGE_IMPORT\n"
    "except ModuleNotFoundError as exc:\n"
    '    if exc.name != "megatron.bridge.models.qwen.qwen_provider":\n'
    "        raise\n"
)

for p in CANDIDATES:
    if not p.exists():
        continue
    src = p.read_text()
    if OLD not in src or "PATCHED_GLM4V_BRIDGE_IMPORT" in src:
        continue
    p.write_text(src.replace(OLD, NEW, 1))
    print(f"Patched {p}")
    break
else:
    print("WARNING: Could not find slime_plugins.megatron_bridge __init__.py to patch")
