from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_4B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4B_Recipe(MilesRecipe):
    """Qwen3-4B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_4B
    sft_supported: ClassVar[bool] = True

    miles_model_name: str = "qwen3-4B"
    recompute_granularity: str | None = "full"

    max_tokens_per_gpu: MicroBatchSize = 8192

    lr: float = 5e-7
