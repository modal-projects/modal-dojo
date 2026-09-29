"""Qwen3.6-35B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_6_response


class Qwen3_6_35B(HFModelConfiguration):
    """Alibaba Qwen3.6-35B-A3B model."""

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.6-35B-A3B"
    architecture = ModelArchitecture(
        num_layers=40,
        hidden_size=2048,
        ffn_hidden_size=512,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=2,
        kv_channels=256,
        vocab_size=248320,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        num_experts=256,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_grouped_gemm=True,
        moe_shared_expert_gate=True,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_token_drop_policy="probs",
        moe_router_dtype="fp32",
        moe_permute_fusion=True,
        moe_aux_loss_coeff=0,
        megatron_spec=["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        megatron_model_type="qwen3.5-35B-A3B",
        apply_layernorm_1p=True,
        use_gated_attention=True,
        attention_output_gate=True,
        use_rotary_position_embeddings=True,
        rotary_base=10000000,
        rotary_percent=0.25,
    )
