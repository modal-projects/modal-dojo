"""Qwen3-30B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_30B(HFModelConfiguration):
    """Alibaba Qwen3-30B-A3B MoE model with 30B total and 3B active parameters."""

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-30B-A3B"
    architecture = ModelArchitecture(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
    )
