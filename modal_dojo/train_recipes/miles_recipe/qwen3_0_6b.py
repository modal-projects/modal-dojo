from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_0_6B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_0_6B_Recipe(MilesRecipe):
    """Qwen3-0.6B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_0_6B
    sft_supported: ClassVar[bool] = True

    miles_model_name: str = "qwen3-0.6B"
    recompute_granularity: str | None = "full"

    sglang_mem_fraction_static: float = 0.78

    max_tokens_per_gpu: MicroBatchSize = 12288

    lr: float = 5e-7
