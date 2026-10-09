from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8B_Recipe(SlimeRecipe):
    """Qwen3-8B recipe."""

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    sglang_mem_fraction_static: float = 0.72
    max_tokens_per_gpu: int = 6144
    lr: float = 5e-7
