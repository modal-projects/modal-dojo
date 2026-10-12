"""
image: slimerl/slime:nightly-dev-20260722a
commit: https://github.com/THUDM/slime/commit/512a1100a0faff379f994caf992b5894f15d47fd
file: slime/slime/utils/data.py::filter_long_prompt
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PATCHED_MULTIMODAL_PROMPT_FILTER"
TARGET = Path("/root/slime/slime/utils/data.py")
ANCHOR = "                multimodal_inputs = process_vision_info(sample.prompt, processor)\n"
REPLACEMENT = (
    f"                # {MARKER}: sample.prompt is already chat-templated text.\n"
    "                multimodal_inputs = sample.multimodal_inputs\n"
)


def _patch_file(path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        print(f"{path.name} already patched for multimodal prompt filter")
        return
    if ANCHOR not in source:
        print(f"WARNING: multimodal prompt filter anchor not found in {path}; skipped")
        return
    path.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"Patched {path} for multimodal prompt filter")


if __name__ == "__main__":
    _patch_file(TARGET)
