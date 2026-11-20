"""Inkling 2 Small: the NVFP4/MXFP8 release, trained as a LoRA through Miles."""

from .base import HFModelConfiguration, parse_inkling_response


class Inkling_2_Small(HFModelConfiguration):
    """594B total / 17B active MoE; architecture comes from Miles' provider."""

    model_name = "thinkingmachines/Inkling-2-Small"
    # Later revisions add a quantization_config key the pinned SGLang rejects.
    revision = "aac974122acd33ddb0d3e1f54433a9c5dd2249ce"
    response_parser = staticmethod(parse_inkling_response)
