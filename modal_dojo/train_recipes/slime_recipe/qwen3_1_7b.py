from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_1_7B_Recipe(SlimeRecipe):
    """Qwen3-1.7B recipe."""

    sglang_mem_fraction_static: float = 0.78
    max_tokens_per_gpu: int = 12288
    lr: float = 5e-7
