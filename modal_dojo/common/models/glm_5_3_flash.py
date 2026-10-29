"""GLM-5.3-Flash's custom KDA/DSA/mHC architecture is supplied by Miles."""

from .base import HFModelConfiguration


class GLM_5_3_Flash(HFModelConfiguration):
    """GLM-5.3-Flash hybrid MoE; the Miles preset trains its text backbone."""

    model_name = "zai-org/GLM-5.3-Flash"
    revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


class GLM_5_3_Flash_LoRA(GLM_5_3_Flash):
    """GLM-5.3-Flash with the Miles native LoRA preset."""
