"""Avoid an FLA 0.4.2 KDA backward autotuning crash on H200/Triton 3.7.

The original search hits a CUDA illegal memory access at BK32/BV64/W4/S2
on packed GLM head dimensions. Use the numerically checked conservative
configuration on Hopper, only in the GLM bridge LoRA image.
"""

from pathlib import Path

MARKER = "PATCHED_TRAINING_GYM_GLM53_LORA_KDA_BACKWARD"
ORIGINAL = """    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'TRANSPOSE_STATE'],"""
REPLACEMENT = """    configs=([
        triton.Config({'BK': 32, 'BV': 32}, num_warps=4, num_stages=1)
    ] if IS_NVIDIA_HOPPER else [
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ]),
    key=['BT', 'TRANSPOSE_STATE'],"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(ORIGINAL) != 1:
        raise ValueError("GLM LoRA KDA backward autotuner changed")
    return f"# {MARKER}\n" + source.replace(ORIGINAL, REPLACEMENT, 1)


def main() -> None:
    from importlib.util import find_spec

    path = Path(find_spec("fla").origin).parent / "ops/kda/chunk_bwd.py"
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
