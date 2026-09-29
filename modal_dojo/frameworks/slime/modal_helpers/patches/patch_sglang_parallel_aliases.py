"""Bridge SGLang's legacy parallel-size names to Slime's new validator names.

The pinned Slime image combines a validator that reads
``sglang_{data,pipeline,expert}_parallel_size`` with an SGLang parser that still
produces ``sglang_{dp,pp,ep}_size``. Populate the new names before validation so
the image's bundled versions remain compatible.
"""

from __future__ import annotations

from pathlib import Path


MARKER = "PATCHED_SGLANG_PARALLEL_SIZE_ALIASES"
UPSTREAM_ALIASES = """\
    for short, long in (
        ("sglang_dp_size", "sglang_data_parallel_size"),
        ("sglang_pp_size", "sglang_pipeline_parallel_size"),
        ("sglang_ep_size", "sglang_expert_parallel_size"),
    ):
        if hasattr(args, long):
            setattr(args, short, getattr(args, long))
        else:
            setattr(args, long, getattr(args, short))
"""
ANCHOR = """\
def validate_args(args):
    args.sglang_dp_size = args.sglang_data_parallel_size
"""
REPLACEMENT = f"""\
def validate_args(args):
    # {MARKER}: bundled SGLang still emits the legacy DP/PP/EP names.
    if not hasattr(args, "sglang_data_parallel_size"):
        args.sglang_data_parallel_size = args.sglang_dp_size
    if not hasattr(args, "sglang_pipeline_parallel_size"):
        args.sglang_pipeline_parallel_size = args.sglang_pp_size
    if not hasattr(args, "sglang_expert_parallel_size"):
        args.sglang_expert_parallel_size = args.sglang_ep_size

    args.sglang_dp_size = args.sglang_data_parallel_size
"""


def _patch_file(path: Path) -> None:
    source = path.read_text()
    if UPSTREAM_ALIASES in source:
        print(f"{path.name} already supports SGLang parallel-size aliases upstream")
        return
    if MARKER in source:
        print(f"{path.name} already patched for SGLang parallel-size aliases")
        return
    if ANCHOR not in source:
        raise RuntimeError(
            f"Could not find SGLang parallel-size validation anchor in {path}"
        )
    path.write_text(source.replace(ANCHOR, REPLACEMENT, 1))
    print(f"Patched {path} with SGLang parallel-size aliases")


def main() -> None:
    _patch_file(Path("/root/slime/slime/backends/sglang_utils/arguments.py"))


if __name__ == "__main__":
    main()
