"""Compat shim: ship + wire a Megatron->HF converter for Qwen3-VL into slime.

slime's MB->HF tool routes a model name containing "qwen3" (but not "qwen3vl") to
convert_qwen2_to_hf, which can't map the vision tower ("Unknown parameter name:
...vision_model..."). Add a qwen3_vl converter and route it before the generic
qwen3 branch. The LLM is the standard Qwen3 decoder nested under language_model.*,
and the frozen vision tower is an identity passthrough (megatron vision_model.* ->
HF model.visual.*): frozen during RL, its weights equal the base HF weights, so a
pure rename is exact.

Idempotent. Run at image build:  python patch_qwen3_vl_export.py
"""

import pathlib

_CONVERTER = r'''"""Megatron -> HF converter for Qwen3-VL (shipped into slime's megatron_to_hf).

slime dispatches a bare "qwen3" name to convert_qwen2_to_hf, which can't map
Qwen3-VL's vision tower. The LLM is the standard Qwen3 decoder nested under
language_model.*, and the frozen vision tower is an identity passthrough (megatron
vision_model.* -> HF model.visual.*).
"""
import re

import torch


def convert_qwen3vl_to_hf(args, name, param):
    # Strip the language_model. nesting to expose the standard megatron decoder
    # names; emit HF names back into the VL model.* namespace.
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    # Frozen vision tower: identity passthrough (rename only).
    if name.startswith("module.module.vision_model."):
        hf = "model.visual." + name[len("module.module.vision_model.") :]
        return [(hf, param)]

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx, rest = match.groups()
        base = f"model.language_model.layers.{layer_idx}"
        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]
        if rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q, k, v = torch.split(param, [value_num_per_group, 1, 1], dim=1)
            return [
                (f"{base}.self_attn.q_proj.weight", q.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.k_proj.weight", k.reshape(-1, args.hidden_size)),
                (f"{base}.self_attn.v_proj.weight", v.reshape(-1, args.hidden_size)),
            ]
        if rest == "mlp.linear_fc1.weight":
            gate, up = param.chunk(2, dim=0)
            return [(f"{base}.mlp.gate_proj.weight", gate), (f"{base}.mlp.up_proj.weight", up)]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]
        if rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        if rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]
    raise ValueError(f"Unknown parameter name: {name}")
'''


def main() -> None:
    base = pathlib.Path("/root/slime/slime/backends/megatron_utils/megatron_to_hf")
    if not base.exists():
        print("compat: slime megatron_to_hf not found at", base)
        return
    (base / "qwen3_vl.py").write_text(_CONVERTER)
    init = base / "__init__.py"
    s = init.read_text()
    if "from .qwen3_vl import convert_qwen3vl_to_hf" not in s:
        if "from .qwen2 import convert_qwen2_to_hf\n" not in s:
            print(
                "compat: WARNING - megatron_to_hf __init__ shape changed; skipping export"
            )
            return
        s = s.replace(
            "from .qwen2 import convert_qwen2_to_hf\n",
            "from .qwen2 import convert_qwen2_to_hf\nfrom .qwen3_vl import convert_qwen3vl_to_hf\n",
            1,
        )
    if '"qwen3vl"' in s or "'qwen3vl'" in s:
        init.write_text(s)
        print("compat: qwen3_vl HF converter shipped; dispatch already wired")
        return
    dispatch_anchor = (
        '    elif "qwen2" in model_name or "qwen3" in model_name:\n'
        "        converted_named_tensors = convert_qwen2_to_hf(args, name, param)\n"
    )
    if dispatch_anchor not in s:
        print(
            "compat: WARNING - megatron_to_hf dispatch shape changed; skipping export"
        )
        return
    s = s.replace(
        dispatch_anchor,
        '    elif "qwen3vl" in model_name or "qwen3_vl" in model_name:\n'
        "        converted_named_tensors = convert_qwen3vl_to_hf(args, name, param)\n"
        + dispatch_anchor,
        1,
    )
    init.write_text(s)
    print("compat: wired qwen3_vl HF converter into slime megatron_to_hf")


if __name__ == "__main__":
    main()
