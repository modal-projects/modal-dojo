"""Preserve explicit context communication in the pinned Miles bridge provider."""

from pathlib import Path

MARKER = "# MODAL_DOJO_BEAM_CP_RUNTIME"
ANCHOR = "    provider.context_parallel_size = args.context_parallel_size\n"
INSERT = """    # MODAL_DOJO_BEAM_CP_RUNTIME
    if getattr(args, "cp_comm_type", None) is not None:
        cp_comm_type = args.cp_comm_type
        # CLI nargs produces a singleton list; MCore expects a scalar for all layers.
        if isinstance(cp_comm_type, (list, tuple)) and len(cp_comm_type) == 1:
            cp_comm_type = cp_comm_type[0]
        provider.cp_comm_type = cp_comm_type
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(ANCHOR) != 1:
        raise RuntimeError("Pinned Miles bridge CP anchor changed")
    return source.replace(ANCHOR, ANCHOR + INSERT)


if __name__ == "__main__":
    path = Path("/root/miles/miles/backends/megatron_utils/model_provider.py")
    path.write_text(patch_source(path.read_text()))
    print("Beam bridge runtime patch: cp_comm_type propagated from args")
