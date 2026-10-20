from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.slime_recipe.recipe import (
    SlimeLossMaskType,
    SlimeRecipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_2B_Recipe(SlimeRecipe):
    """Qwen3.5-2B recipe."""

    loss_mask_type: SlimeLossMaskType = "qwen3_5"
    sglang_mem_fraction_static: float = 0.78
    attention_backend: str = "flash"
    max_tokens_per_gpu: MicroBatchSize = 12288
    lr: float = 5e-7
