"""Thin wrapper around slime's convert_torch_dist_to_hf.py for Modal volumes.

Transparent pass-through to the upstream script.
"""

from __future__ import annotations

import os
import runpy

_UPSTREAM = "/root/slime/tools/convert_torch_dist_to_hf.py"


def main() -> None:
    if not os.path.exists(_UPSTREAM):
        raise FileNotFoundError(f"Upstream conversion script not found: {_UPSTREAM}")
    runpy.run_path(_UPSTREAM, run_name="__main__")


if __name__ == "__main__":
    main()
