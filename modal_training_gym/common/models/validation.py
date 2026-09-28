"""Model configs supported by the CI validation run.

One registry for every framework. Each entry names the model, its
``ModelConfig``, and the framework whose base recipe trains it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..framework import Framework
from .base import ModelConfig
from .deepseek_v41_flash import DeepSeek_V4_1_Flash
from .gemma4_26b_a4b import Gemma4_26B_A4B
from .glm_4_7 import GLM_4_7
from .inkling_small import Inkling_Small, Inkling_Small_LoRA
from .moonlight_16b_a3b_instruct import Moonlight_16B_A3B_Instruct
from .qwen3_0_6b import Qwen3_0_6B
from .qwen3_1_7b import Qwen3_1_7B
from .qwen3_4b import Qwen3_4B
from .qwen3_5_0_8b import Qwen3_5_0_8B
from .qwen3_5_2b import Qwen3_5_2B
from .qwen3_5_4b import Qwen3_5_4B
from .qwen3_5_9b import Qwen3_5_9B
from .qwen3_6_27b import Qwen3_6_27B
from .qwen3_6_35b import Qwen3_6_35B
from .qwen3_8_27b import Qwen3_8_27B
from .qwen3_8b import Qwen3_8B
from .qwen3_asr_1_7b import Qwen3_ASR_1_7B
from .qwen3_vl_8b import Qwen3_VL_8B


@dataclass(frozen=True)
class _ValidationConfig:
    """One model/framework pair the validation harness runs."""

    # Short name used as ``check --model`` and as the CI matrix entry.
    name: str
    model_config: type[ModelConfig]
    # Which framework's ``get_base_recipe`` trains this model.
    framework: Framework
    loss_type: str = "policy_loss"

    @property
    def model_name(self) -> str:
        """The Hugging Face repository id."""
        return self.model_config.model_name

    @classmethod
    def select(cls, framework: Framework | None = None) -> list["_ValidationConfig"]:
        """Registry entries, name-sorted, optionally limited to one framework."""
        return sorted(
            (
                config
                for config in VALIDATION_CONFIGS
                if framework is None or config.framework is framework
            ),
            key=lambda config: config.name,
        )

    @classmethod
    def find(cls, name: str) -> "_ValidationConfig":
        """Look up an entry by short name or HF repo id, case-insensitively."""
        wanted = name.strip().lower()
        matches = [
            config
            for config in VALIDATION_CONFIGS
            if wanted == config.name.lower()
            or (
                wanted == config.model_name.lower()
                and config.loss_type == "policy_loss"
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            choices = ", ".join(sorted(config.name for config in matches))
            raise ValueError(f"ambiguous model {name!r}; use one of: {choices}")
        available = ", ".join(config.name for config in cls.select())
        raise ValueError(f"unknown model {name!r}; available: {available}")


VALIDATION_CONFIGS: set[_ValidationConfig] = {
    _ValidationConfig("GLM-4.7", GLM_4_7, Framework.SLIME),
    _ValidationConfig("Qwen3-0.6B", Qwen3_0_6B, Framework.SLIME),
    _ValidationConfig("Qwen3-1.7B", Qwen3_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-4B", Qwen3_4B, Framework.SLIME),
    _ValidationConfig("Qwen3-8B", Qwen3_8B, Framework.SLIME),
    _ValidationConfig("Qwen3-ASR-1.7B", Qwen3_ASR_1_7B, Framework.SLIME),
    _ValidationConfig("Qwen3-VL-8B-Instruct", Qwen3_VL_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-0.8B", Qwen3_5_0_8B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-2B", Qwen3_5_2B, Framework.SLIME),
    _ValidationConfig("Qwen3.5-4B", Qwen3_5_4B, Framework.SLIME),
    _ValidationConfig(
        "Qwen3.5-4B-SFT", Qwen3_5_4B, Framework.SLIME, loss_type="sft_loss"
    ),
    _ValidationConfig("Qwen3.5-4B-Miles", Qwen3_5_4B, Framework.MILES),
    _ValidationConfig("Qwen3.5-9B", Qwen3_5_9B, Framework.SLIME),
    _ValidationConfig("Qwen3.6-27B", Qwen3_6_27B, Framework.SLIME),
    _ValidationConfig("Qwen3.6-35B-A3B", Qwen3_6_35B, Framework.SLIME),
    _ValidationConfig("Qwen3.8-27B", Qwen3_8_27B, Framework.SLIME),
    _ValidationConfig(
        "Moonlight-16B-A3B-Instruct",
        Moonlight_16B_A3B_Instruct,
        Framework.MILES,
    ),
    _ValidationConfig(
        "Moonlight-16B-A3B-Instruct-SFT",
        Moonlight_16B_A3B_Instruct,
        Framework.MILES,
        loss_type="sft_loss",
    ),
    _ValidationConfig("Gemma-4-26B-A4B-it", Gemma4_26B_A4B, Framework.MILES),
    _ValidationConfig("Inkling-Small", Inkling_Small, Framework.MILES),
    _ValidationConfig("Inkling-Small-LoRA", Inkling_Small_LoRA, Framework.MILES),
    _ValidationConfig("DeepSeek-V4.1-Flash", DeepSeek_V4_1_Flash, Framework.MILES),
}
