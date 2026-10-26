from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_5_2B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_2B_Recipe(MilesRecipe):
    """Qwen3.5-2B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_5_2B
    sft_supported: ClassVar[bool] = False

    recompute_granularity: str | None = "full"

    sglang_mem_fraction_static: float = 0.78

    attention_backend: str = "flash"

    max_tokens_per_gpu: MicroBatchSize = 12288

    lr: float = 5e-7
