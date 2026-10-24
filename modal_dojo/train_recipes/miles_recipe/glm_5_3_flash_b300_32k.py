"""B300 counterpart of the native GLM-5.3-Flash 32k LoRA preset."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.miles_recipe.glm_5_3_flash_lora import (
    GLM_5_3_Flash_LoRA_Recipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_LoRA_B300_32k_Recipe(GLM_5_3_Flash_LoRA_Recipe):
    """Rank-16 native LoRA on 8 B300s; GPU validation is pending."""

    gpu_type: str = "B300"
    actor_num_nodes: int = 1
    expert_model_parallel_size: int = 8
    sglang_max_running_requests: int = 32
