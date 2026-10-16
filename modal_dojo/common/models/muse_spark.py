"""Muse Spark Open: the BF16 omni checkpoint, with text RL through Miles."""

from .base import HFModelConfiguration


class Muse_Spark_Open(HFModelConfiguration):
    """816B total / 42B active MoE; architecture comes from Miles' provider."""

    model_name = "meta-models/Muse-Spark-Open-816B-A42B"
    revision = "0fb02478b02565b2ce7fc57e6051c91172774ac9"
