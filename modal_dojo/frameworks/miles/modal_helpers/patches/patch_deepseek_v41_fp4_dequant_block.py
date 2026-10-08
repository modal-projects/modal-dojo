"""Patch sglang's FP4->FP8 expert dequant to emit the checkpoint's block size.

With ``SGLANG_DSV4_FP4_DEQUANT=1`` (the Hopper path for DeepSeek-V4 checkpoints
whose routed experts ship as packed e2m1), ``Fp8MoEMethod`` casts every expert to
e4m3 at load through ``cast_e2m1fn_to_e4m3fn``. sgl-project/sglang#38798 hardcodes
that cast to 128x128 scale blocks, which matched DeepSeek-V4's
``weight_block_size=[128, 128]``. DeepSeek-V4.1 declares ``[32, 32]``: the MoE
runner keeps consuming ``quant_config.weight_block_size``, so the Triton kernel
asserts ``cdiv(N, 32) == N // 128`` on the first forward, and Miles' weight sync
(which quantizes trainer experts at the config's block size) would push
``(N/32, K/32)`` scales into ``(N/128, K/128)`` parameters. Let the cast take the
target block size and pass the config's, so the dequantized experts, the runner,
and the trainer-side quantizer all agree.

Executed at image-build time via ``python3 <this file>``.

Image: radixark/miles:dsv41-h200-ea751aac8
SGLang: 7e74b31b2668934cf89dbe15188a010fda381aba
"""

import pathlib

MARKER = "PATCHED_DEEPSEEK_V41_FP4_DEQUANT_BLOCK"

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/fp8.py"
)

OLD_SIGNATURE = """def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
"""
NEW_SIGNATURE = f"""def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor, fp8_block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:  # {MARKER}
"""

OLD_BLOCK = """    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
"""
NEW_BLOCK = """    fp4_block_size = 32
    assert fp8_block_size % fp4_block_size == 0
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
"""

OLD_CALL = """                    num_experts = weight_param.shape[0]
                    new_weights = []
                    new_scales = []
                    for e in range(num_experts):
                        w, s = cast_e2m1fn_to_e4m3fn(
                            weight_param.data[e], scale_param.data[e]
                        )
"""
NEW_CALL = """                    num_experts = weight_param.shape[0]
                    new_weights = []
                    new_scales = []
                    block_n, block_k = self.quant_config.weight_block_size or [128, 128]
                    assert block_n == block_k, self.quant_config.weight_block_size
                    for e in range(num_experts):
                        w, s = cast_e2m1fn_to_e4m3fn(
                            weight_param.data[e],
                            scale_param.data[e],
                            fp8_block_size=block_n,
                        )
"""

# The H200 image also has a shared dequant helper used by other MoE backends.
# Keep both call sites consistent if the caller selects one of those backends.
OLD_HELPER_CALL = """            num_experts = weight_param.shape[0]
            new_weights = []
            new_scales = []
            for e in range(num_experts):
                w, s = cast_e2m1fn_to_e4m3fn(weight_param.data[e], scale_param.data[e])
"""
NEW_HELPER_CALL = """            num_experts = weight_param.shape[0]
            new_weights = []
            new_scales = []
            block_n, block_k = self.quant_config.weight_block_size or [128, 128]
            assert block_n == block_k, self.quant_config.weight_block_size
            for e in range(num_experts):
                w, s = cast_e2m1fn_to_e4m3fn(
                    weight_param.data[e], scale_param.data[e], fp8_block_size=block_n
                )
"""


def patch_source(src: str) -> str:
    if MARKER in src:
        return src
    replacements = (
        (OLD_SIGNATURE, NEW_SIGNATURE),
        (OLD_BLOCK, NEW_BLOCK),
        (OLD_CALL, NEW_CALL),
        (OLD_HELPER_CALL, NEW_HELPER_CALL),
    )
    for old, _ in replacements:
        if src.count(old) != 1:
            raise ValueError(
                "DeepSeek-V4.1 FP4 dequant block patch did not match; re-check "
                "cast_e2m1fn_to_e4m3fn and both Fp8MoEMethod dequant paths."
            )
    for old, new in replacements:
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    TARGET.write_text(patch_source(TARGET.read_text()))
    print("Patched FP4 expert dequant block size in both Fp8MoEMethod paths")
