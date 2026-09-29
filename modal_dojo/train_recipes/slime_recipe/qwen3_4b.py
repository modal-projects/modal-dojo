from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4B_Recipe(SlimeRecipe):
    """Qwen3-4B recipe."""

    max_tokens_per_gpu: int = 8192
    lr: float = 5e-7
