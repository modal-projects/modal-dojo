"""Qwen3-0.6B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_0_6B(HFModelConfiguration):
    """Alibaba Qwen3-0.6B model."""

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-0.6B"
    architecture = ModelArchitecture(
        num_layers=28,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        rotary_base=1000000,
    )
