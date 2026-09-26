from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_9B_Recipe(SlimeRecipe):
    """Qwen3.5-9B recipe."""

    loss_mask_type: Literal["qwen", "qwen3", "qwen3_5", "distill_qwen"] = "qwen3_5"
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    sglang_mem_fraction_static: float = 0.6
    attention_backend: str = "flash"
    max_tokens_per_gpu: int = 6144
    lr: float = 5e-7
