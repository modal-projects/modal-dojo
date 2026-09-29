"""Qwen3.5-9B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_6_response


class Qwen3_5_9B(HFModelConfiguration):
    """Alibaba Qwen3.5-9B model."""

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.5-9B"
    architecture = ModelArchitecture(
        num_layers=32,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=256,
        vocab_size=248320,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        megatron_spec=["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        apply_layernorm_1p=True,
        use_gated_attention=True,
        attention_output_gate=True,
        use_rotary_position_embeddings=True,
        rotary_base=10000000,
        rotary_percent=0.25,
    )
