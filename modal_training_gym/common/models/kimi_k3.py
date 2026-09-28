"""Kimi-K3 model spec.

KDA/MLA layers require upstream's model script rather than ModelArchitecture.
Miles converts native MXFP4 experts to BF16 for training; rollout uses MXFP4.
"""

from __future__ import annotations

from .base import HFModelConfiguration


class Kimi_K3(HFModelConfiguration):
    """Moonshot Kimi-K3 hybrid KDA/MLA MoE model, 93 layers and 896 routed experts."""

    model_name = "moonshotai/Kimi-K3"
