from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_8_27B
from modal_dojo.train_recipes.spindle_recipe.recipe import SpindleRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8_27B_Spindle_Recipe(SpindleRecipe):
    """Qwen3.8-27B LoRA at 32K context on Spindle.

    Mirrors Spindle's ``qwen38_27b_lora_16k`` deployment (8xH200 trainer, TP4,
    rank-32 LoRA) with the context length raised to 32K.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_8_27B

    miles_model_name: str = "qwen3.8-27B"
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    rollout_max_response_len: int = 16384
    environment: dict = field(
        default_factory=lambda: {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
