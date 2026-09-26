from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_0_8B_Recipe(SlimeRecipe):
    """Qwen3.5-0.8B recipe."""

    loss_mask_type: Literal["qwen", "qwen3", "qwen3_5", "distill_qwen"] = "qwen3_5"
    sglang_mem_fraction_static: float = 0.78
    attention_backend: str = "flash"
    max_tokens_per_gpu: int = 12288
    lr: float = 5e-7
