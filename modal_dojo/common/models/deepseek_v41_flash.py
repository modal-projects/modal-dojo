"""DeepSeek-V4.1-Flash model spec as a concrete HFModelConfiguration subclass.

DeepSeek-V4.1-Flash is a 40-layer, 5120-hidden MoE (384 routed experts, top-6)
built on DeepSeek's sparse attention line: a DSA indexer selects the KV entries
each query attends to, and V4.1 adds Engram — a hashed n-gram memory whose
tables live outside the transformer stack.

Neither the indexer nor Engram is representable as a ``ModelArchitecture``, so
``architecture`` is left ``None`` and the Miles recipe renders upstream's
``scripts/models/deepseek-v4.1.py`` via ``miles_model_name`` — including its
``--spec miles_plugins.models.deepseek_v41.deepseek_v41 get_dsv41_spec``. Same
arrangement as ``Qwen3_5_4B_Miles``.

The released checkpoint is fp8; Miles converts it to a bf16 torch_dist
checkpoint before training.
"""

from __future__ import annotations

from .base import HFModelConfiguration


class DeepSeek_V4_1_Flash(HFModelConfiguration):
    """DeepSeek-V4.1-Flash sparse-attention MoE model, 40 layers and 384 routed experts."""

    model_name = "deepseek-ai/DeepSeek-V4.1-Flash"
