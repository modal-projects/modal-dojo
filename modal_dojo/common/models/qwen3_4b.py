"""Qwen3-4B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_4B(HFModelConfiguration):
    """Alibaba Qwen3-4B model."""

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-4B"
    architecture = ModelArchitecture(
        num_layers=36,
        hidden_size=2560,
        ffn_hidden_size=9728,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        rotary_base=1000000,
    )
