"""Validate that every file mounting a volume at /root/.cache/huggingface also
clears the directory during image build.

Modal volumes cannot mount on non-empty paths.  Several base images (slime,
sglang, vllm) ship with files under /root/.cache/huggingface after pip install,
so every image that mounts the HF cache volume must include a
``rm -rf /root/.cache/huggingface`` (or equivalent) in its build commands.

This script greps the source tree for the pattern and flags any file that
mounts the volume but omits the cleanup — the exact bug that caused
crashlooping serve apps with the error:

    Runner failed with exception: cannot mount volume on non-empty path:
    "/root/.cache/huggingface"

Usage:
    uv run scripts/test_hf_cache_volume_safety.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "modal_dojo"

VOLUME_MOUNT_PATTERN = re.compile(
    r"""['"]\/root\/\.cache\/huggingface['"]"""
    r""".*(?:Volume|vol|hf_cache)""",
    re.IGNORECASE,
)

VOLUME_DICT_PATTERN = re.compile(
    r"""/root/\.cache/huggingface""",
)

CLEANUP_PATTERN = re.compile(
    r"""rm\s+-rf\s+/root/\.cache/huggingface""",
)


VOLUME_MOUNT_RE = re.compile(
    r"""['"]\/root\/\.cache\/huggingface['"]\s*:\s*\w*[Vv]ol""",
)


def _mounts_hf_cache_volume(source: str) -> bool:
    return bool(VOLUME_MOUNT_RE.search(source))


def _has_cache_cleanup(source: str) -> bool:
    return bool(CLEANUP_PATTERN.search(source))


def check_files() -> list[str]:
    failures = []
    for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = py_file.read_text()
        if _mounts_hf_cache_volume(source) and not _has_cache_cleanup(source):
            failures.append(str(py_file.relative_to(PACKAGE_ROOT.parent)))
    return failures


def main() -> int:
    failures = check_files()
    if failures:
        print("FAIL: These files mount /root/.cache/huggingface as a volume")
        print("      but do NOT clear the directory during image build:\n")
        for f in failures:
            print(f"  - {f}")
        print(
            '\nAdd .run_commands("rm -rf /root/.cache/huggingface") to the'
            " image build chain before the volume is mounted at runtime."
        )
        return 1
    else:
        print(
            "OK: All files that mount /root/.cache/huggingface also clear"
            " the directory during image build."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
