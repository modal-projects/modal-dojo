"""Inkling-Small model spec as a concrete HFModelConfiguration subclass.

Inkling-Small is Thinking Machines Lab's 276 B-total / 12 B-active, 42-layer
multimodal MoE (256 routed experts + a shared-expert sink, top-6 sigmoid routing).
It is not an ordinary Megatron decoder: no positional embeddings (a learned
relative-position bias instead), a residual short causal convolution on the K/V
streams and on the attention and MoE outputs, and a mix of sliding-window and
full-attention layers.

That architecture is not representable as a ``ModelArchitecture``, so
``architecture`` is deliberately left ``None`` and the Miles recipes source
upstream's ``scripts/models/inkling-small.sh`` via ``miles_model_script``
instead — including its ``--custom-model-provider-path``. Populating
``architecture`` here would make ``get_checkpoint_conversion_policy`` emit arch
flags *on top of* the sourced ``${MODEL_ARGS[@]}`` at conversion time. Same
arrangement as ``Kimi_K2_5``.

The released checkpoint is the multimodal one (``model_type: inkling_mm_model``,
image-text-to-text and audio-text-to-text): the vision and audio towers ship in
the same repo and are loaded frozen when the recipe selects the multimodal
provider.
"""

from __future__ import annotations

from .base import HFModelConfiguration, parse_inkling_response


class Inkling_Small(HFModelConfiguration):
    """Thinking Machines Lab Inkling-Small MoE model with 276B total and 12B active parameters."""

    model_name = "thinkingmachines/Inkling-Small"
    response_parser = staticmethod(parse_inkling_response)


class Inkling_Small_LoRA(Inkling_Small):
    """Selects ``Inkling_Small_LoRA_Recipe``."""
