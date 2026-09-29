"""Patch sglang's block-FP8 Triton GEMM to dispatch the Hopper-tuned kernel.

The sglang tree radixark ships for V4.1 (see patch_deepseek_v41_sglang_tree)
carries sgl-project/sglang#38798's ``_w8a8_block_fp8_matmul_hopper`` kernel and
its H200 ``block_shape=[32, 32]`` tuning tables, whose entries add ``SWAP_AB`` and
``SPLIT_K``. The merge dropped the dispatch in ``w8a8_block_fp8_matmul_triton``
that routes such configs to that kernel, so on H200 every dense FP8 projection
forwards ``SWAP_AB`` into the generic kernel and Triton rejects it:

    KeyError: 'Keyword argument SWAP_AB was specified but unrecognised'

Restore the PR's dispatch: on sm90 with a tuned config, run the Hopper kernel
over a ``(tiles, SPLIT_K)`` grid into fp32 partials and reduce them with the
tree's ``_reduce_block_fp8_split_k``; otherwise keep the generic path.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib

MARKER = "PATCHED_DEEPSEEK_V41_FP8_HOPPER_GEMM"

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py"
)

OLD_DISPATCH = """    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    kernel = select_w8a8_block_fp8_matmul_kernel(M, N, config)

    kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    return C
"""

NEW_DISPATCH = f"""    kernel = select_w8a8_block_fp8_matmul_kernel(M, N, config)

    # {MARKER}
    hopper_tuned = get_device_sm() == 90 and (
        config.get("SWAP_AB", False) or config.get("SPLIT_K", 1) > 1
    )
    if hopper_tuned:
        kernel = _w8a8_block_fp8_matmul_hopper
    else:
        config = {{k: v for k, v in config.items() if k not in ("SWAP_AB", "SPLIT_K")}}
    split_k = config.get("SPLIT_K", 1) if hopper_tuned else 1
    if split_k > 1:
        assert split_k & (split_k - 1) == 0
        partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
    else:
        partials = C

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        blocks = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N, META["BLOCK_SIZE_N"]
        )
        return (blocks, split_k) if hopper_tuned else (blocks,)

    kernel[grid](
        A,
        B,
        partials,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    if split_k > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](
            partials, C, M * N, split_k, 256
        )

    return C
"""

OLD_IMPORT = """from sglang.srt.utils import (
    ceil_align,
    get_bool_env_var,
    get_device_core_count,
    get_device_name,
"""
NEW_IMPORT = """from sglang.srt.utils import (
    ceil_align,
    get_bool_env_var,
    get_device_core_count,
    get_device_name,
    get_device_sm,
"""

if not TARGET.exists():
    print(f"{TARGET} not found; skipping DeepSeek-V4.1 FP8 Hopper GEMM patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("DeepSeek-V4.1 FP8 Hopper GEMM patch already applied")
    raise SystemExit(0)

for old in (OLD_IMPORT, OLD_DISPATCH):
    if src.count(old) != 1:
        raise SystemExit(
            "DeepSeek-V4.1 FP8 Hopper GEMM patch did not match; sglang's "
            "kernels/ops/quantization/fp8_kernel.py has changed. Re-check "
            "w8a8_block_fp8_matmul_triton before shipping."
        )
for needed in ("def _w8a8_block_fp8_matmul_hopper(", "def _reduce_block_fp8_split_k("):
    if needed not in src:
        raise SystemExit(f"DeepSeek-V4.1 FP8 Hopper GEMM patch: {needed} missing")

src = src.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_DISPATCH, NEW_DISPATCH, 1)
TARGET.write_text(src)
print("Patched sglang fp8_kernel.py: Hopper-tuned block-FP8 GEMM dispatch restored")
