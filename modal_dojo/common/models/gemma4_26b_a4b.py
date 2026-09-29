"""Gemma-4-26B-A4B model spec as a concrete HFModelConfiguration subclass."""

from __future__ import annotations

from .base import HFModelConfiguration, ModelArchitecture, parse_gemma4_response


class Gemma4_26B_A4B(HFModelConfiguration):
    """Google Gemma-4-26B-A4B-it multimodal MoE model with 26B total and 4B active parameters."""

    model_name = "google/gemma-4-26B-A4B-it"
    response_parser = staticmethod(parse_gemma4_response)

    architecture = ModelArchitecture(
        # text_config from config.json. The recipe sets ``miles_model_script``, so
        # these are not emitted as flags, but they drive the expert-parallel
        # validator; keep them in step with that script.
        num_layers=30,
        hidden_size=2816,
        ffn_hidden_size=2112,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=256,
        vocab_size=262144,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=False,  # GeGLU: set by the layer spec, not a Megatron flag
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=False,
        num_experts=128,
        moe_ffn_hidden_size=704,
        moe_grouped_gemm=True,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        use_rotary_position_embeddings=True,
        # rope_theta is nested per attention type; this is the global-attention one.
        rotary_base=1000000,
        rotary_percent=1.0,
    )
