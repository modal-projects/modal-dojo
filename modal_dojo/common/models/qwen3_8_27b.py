"""Qwen3.8-27B model configuration."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_6_response


class Qwen3_8_27B(HFModelConfiguration):
    """Alibaba Qwen3.8-27B model."""

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.8-27B"
    architecture = ModelArchitecture(
        num_layers=64,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=24,
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
        megatron_model_type="qwen3.5-27B",
        apply_layernorm_1p=True,
        use_gated_attention=True,
        attention_output_gate=True,
        use_rotary_position_embeddings=True,
        rotary_base=10000000,
        rotary_percent=0.25,
    )
