"""Patch PyTorch's DefaultSavePlanner to skip global plan validation.

MoE models with expert parallelism distribute disjoint expert parameters
across EP ranks, which trips PyTorch's plan coverage check during
checkpoint save.  This replaces the hard error with a warning.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

p = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages"
    "/torch/distributed/checkpoint/default_planner.py"
)
if p.exists():
    src = p.read_text()
    old = 'raise ValueError("Failed to validate global plan")'
    if old in src and "PATCHED_GLOBAL_PLAN" not in src:
        new = (
            "import warnings as _w; "
            '_w.warn("Skipped global plan validation (MoE/hybrid model)")  '
            "# PATCHED_GLOBAL_PLAN"
        )
        src = src.replace(old, new, 1)
        p.write_text(src)
        print("Patched default_planner.py to skip global plan validation")
