"""Patch sglang's DeepSeek-V4.1 ``vision_topk`` to feed the routed-experts capturer.

DeepSeek-V4.1 carries a vision-token expert bias (``e_score_correction_bias_vl``),
so ``DeepseekV2MoE.forward`` routes every MoE layer through
``multimodal/dsv41/vl_routing.py::vision_topk`` instead of ``TopK.forward`` /
``select_experts``. That fused path returns ``(weights, indices)`` directly and
skips ``capture_routed_experts_if_allowed``, so with
``--enable-return-routed-experts`` the ``routed_experts`` buffer sglang hands
back to miles is never written and decodes to all zeros. Miles' rollout routing
replay (R3) then replays ``[0, 0, 0, 0, 0, 0]`` for every token; Megatron's
dropless dispatcher sizes the permuted buffer as ``num_tokens * topk`` but the
all-to-all splits come from ``routing_map.sum(0)`` (one hit per token), which
surfaces as ``RuntimeError: Split sizes doesn't match total dim 0 size`` in
``token_dispatcher.token_dispatch``.

Run the same capture hook ``select_experts`` runs, mirroring
``build_precomputed_topk_output``.

Executed at image-build time via ``python3 <this file>``.

Image: radixark/miles:dev-202610082133
SGLang: b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02
"""

import pathlib

MARKER = "PATCHED_DEEPSEEK_V41_VISION_TOPK_CAPTURE"

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/multimodal/dsv41/vl_routing.py"
)

OLD_IMPORT = """from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
)
"""
NEW_IMPORT = f"""from sglang.srt.layers.moe.topk import (  # {MARKER}
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    StandardTopKOutputPacked,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
    capture_routed_experts_if_allowed,
)
"""

OLD_CUDA_RETURN = """        if packed_topk is not None:
            return StandardTopKOutputPacked(weights, indices, logits, packed_topk)
"""
NEW_CUDA_RETURN = """        capture_routed_experts_if_allowed(config, moe.layer_id, indices)
        if packed_topk is not None:
            return StandardTopKOutputPacked(weights, indices, logits, packed_topk)
"""

OLD_NATIVE_RETURN = """        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    return StandardTopKOutput(weights, indices, logits)
"""
NEW_NATIVE_RETURN = """        _zero_topk_weights_padded_region(weights, num_token_non_padded)
    capture_routed_experts_if_allowed(config, moe.layer_id, indices)
    return StandardTopKOutput(weights, indices, logits)
"""


def patch_source(src: str) -> str:
    if MARKER in src:
        return src
    replacements = (
        (OLD_IMPORT, NEW_IMPORT),
        (OLD_CUDA_RETURN, NEW_CUDA_RETURN),
        (OLD_NATIVE_RETURN, NEW_NATIVE_RETURN),
    )
    for old, _ in replacements:
        if src.count(old) != 1:
            raise ValueError(
                "DeepSeek-V4.1 vision_topk capture patch did not match; re-check "
                "whether vision_topk now captures routed experts itself."
            )
    for old, new in replacements:
        src = src.replace(old, new, 1)
    return src


if __name__ == "__main__":
    TARGET.write_text(patch_source(TARGET.read_text()))
    print("Patched vision_topk routed-expert capture, including packed CUDA output")
