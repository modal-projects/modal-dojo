"""Thin wrapper around slime's convert_hf_to_torch_dist.py for Modal volumes.

The upstream script's shutil.move(iter_0000001 -> release) only sees local shards,
which poisons the volume state for multi-node conversions (the rename propagates a
deletion of iter_0000001/ that wipes other nodes' committed shards).

Setting SKIP_RELEASE_RENAME=1 suppresses the rename so all nodes commit
to iter_0000001/ additively. Megatron loads from iter_0000001/ via the
tracker file just fine.

Setting SKIP_PP_AUTOINFLATE=1 disables the conversion script's automatic
inflation of pipeline_model_parallel_size to world_size.  This lets callers
control PP/TP explicitly so the checkpoint layout matches training.

When neither env var is set this wrapper is a transparent pass-through.
"""

from __future__ import annotations

import os

_UPSTREAM = "/root/slime/tools/convert_hf_to_torch_dist.py"


def _load_upstream_source() -> str:
    with open(_UPSTREAM) as f:
        return f.read()


def main() -> None:
    src = _load_upstream_source()
    if os.environ.get("SKIP_RELEASE_RENAME"):
        src = src.replace(
            "shutil.move(source_dir, target_dir)",
            "pass  # SKIP_RELEASE_RENAME",
        )
        src = src.replace(
            'f.write("release")',
            'f.write("1")  # SKIP_RELEASE_RENAME: keep iter_0000001',
        )
    if os.environ.get("SKIP_PP_AUTOINFLATE"):
        src = src.replace(
            "if args.pipeline_model_parallel_size == 1 and world_size > 1:",
            "if False:  # SKIP_PP_AUTOINFLATE",
        )
    exec(
        compile(src, _UPSTREAM, "exec"), {"__name__": "__main__", "__file__": _UPSTREAM}
    )


if __name__ == "__main__":
    main()
