"""Reflection Beam's hybrid attention and MoE semantics are supplied by Miles Bridge."""

from .base import HFModelConfiguration


class Beam_1_0_501B_A23B(HFModelConfiguration):
    """Reflection Beam 1.0: 501B total parameters, 23B active."""

    model_name = "reflection/Beam-1.0-501B-A23B"
    revision = "50a18f8ccf28c75cfdfd9cfe602451c5c81acaef"


class Beam_1_0_501B_A23B_LoRA(Beam_1_0_501B_A23B):
    """Select the rank-16 Bridge LoRA preset."""
