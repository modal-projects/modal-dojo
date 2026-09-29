"""
image: radixark/miles:dev-202608041247
commit: https://github.com/radixark/miles/commit/5c517599fb55c2528a55c749e6eda5f99f51f0ad
file: miles/miles_plugins/models/inkling/layers.py::InklingSelfAttention.sharded_state_dict
"""

from __future__ import annotations

import pathlib
import re

MARKER = "PATCHED_QKVR_CPU_MERGE"

TARGET = pathlib.Path("/root/miles/miles_plugins/models/inkling/layers.py")

OLD_RE = re.compile(
    r"(?P<indent>[ \t]*)def _qkvr_merge\(sub\):\n"
    r"(?P=indent)    return torch\.cat\(list\(sub\), dim=0\)\n"
)


def _replacement(match: re.Match[str]) -> str:
    indent = match.group("indent")
    return (
        f"{indent}def _qkvr_merge(sub):  # {MARKER}\n"
        f"{indent}    host = [p.detach().cpu().contiguous() for p in sub]\n"
        f"{indent}    return torch.cat(host, dim=0)\n"
    )


def apply(target: pathlib.Path = TARGET) -> int:
    if not target.exists():
        print(f"{target} not found; skipping qkvr CPU-merge patch")
        return 0

    src = target.read_text()
    if MARKER in src:
        print("qkvr CPU-merge patch already applied")
        return 0

    patched, n = OLD_RE.subn(_replacement, src, count=1)
    if n != 1:
        raise SystemExit(
            "qkvr CPU-merge patch did not match; miles_plugins Inkling "
            "layers.py _qkvr_merge has changed. Re-check it before shipping."
        )

    target.write_text(patched)
    print(f"Patched {target} _qkvr_merge to CPU-merge")
    return 1


if __name__ == "__main__":
    apply()
