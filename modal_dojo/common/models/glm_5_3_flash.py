"""GLM-5.3-Flash's custom KDA/DSA/mHC architecture is supplied by Miles."""

from .base import HFModelConfiguration


class GLM_5_3_Flash(HFModelConfiguration):
    """GLM-5.3-Flash hybrid MoE; the Miles preset trains its text backbone."""

    model_name = "zai-org/GLM-5.3-Flash"
    revision = "91b69215804f2d2036a8867f6ce44848e278dcda"


class GLM_5_3_Flash_LoRA(GLM_5_3_Flash):
    """GLM-5.3-Flash with the experimental Miles bridge LoRA preset."""
