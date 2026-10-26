from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_8B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8B_Recipe(MilesRecipe):
    """Qwen3-8B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_8B
    sft_supported: ClassVar[bool] = True

    miles_model_name: str = "qwen3-8B"
    recompute_granularity: str | None = "full"

    optimizer_cpu_offload: bool = True

    overlap_cpu_optimizer_d2h_h2d: bool = True

    use_precision_aware_optimizer: bool = True

    sglang_mem_fraction_static: float = 0.72

    max_tokens_per_gpu: MicroBatchSize = 6144

    lr: float = 5e-7
