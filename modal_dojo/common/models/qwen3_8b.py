"""Qwen3-8B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture, parse_qwen3_response


class Qwen3_8B(HFModelConfiguration):
    """Alibaba Qwen3-8B model."""

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-8B"
    architecture = ModelArchitecture(
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        untie_embeddings_and_output_weights=True,
        rotary_base=1000000,
    )
