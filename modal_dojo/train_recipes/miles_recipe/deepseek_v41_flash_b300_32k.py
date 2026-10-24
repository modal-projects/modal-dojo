"""DeepSeek V4.1 Flash B300 presets, sized statically for 32k context."""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash import (
    DeepSeek_V4_1_Flash_Recipe,
)
from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash_lora import (
    DeepSeek_V4_1_Flash_LoRA_Recipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_B300_32k_Recipe(DeepSeek_V4_1_Flash_Recipe):
    """Full-parameter GRPO on 32 B300s; GPU validation is pending."""

    gpu_type: str = "B300"
    actor_num_nodes: int = 4
    expert_model_parallel_size: int = 32
    # ~127 GiB of BF16 experts per rank at EP8 fits a B300 conversion node.
    conversion_expert_model_parallel_size: int = 8
    rollout_max_prompt_len: int = 4096
    rollout_max_response_len: int = 28672
    rollout_max_context_len: int = 32768
    seq_length: int = 32768
    max_tokens_per_gpu: MicroBatchSize = 32768
    sglang_context_length: int = 32768
    sglang_max_running_requests: int = 32
    environment: dict[str, str] = field(
        default_factory=lambda: {
            **DeepSeek_V4_1_Flash_Recipe().environment,
            # Modal's B300 nodes do not have a multi-node NVLink fabric.
            "NCCL_MNNVL_ENABLE": "0",
        }
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_LoRA_B300_32k_Recipe(DeepSeek_V4_1_Flash_LoRA_Recipe):
    """Rank-32 native LoRA on 8 B300s; GPU validation is pending."""

    gpu_type: str = "B300"
    actor_num_nodes: int = 1
    expert_model_parallel_size: int = 8
    conversion_expert_model_parallel_size: int = 8
    sglang_max_running_requests: int = 32
    # One node now holds the whole BF16 base backup plus CPU Engram tables.
    # Reserve 2.75 TiB, leaving room above their ~1.5 TiB static estimate.
    memory: int | tuple[int, int] | None = (2816 * 1024, 2816 * 1024)
    environment: dict[str, str] = field(
        default_factory=lambda: {
            **DeepSeek_V4_1_Flash_LoRA_Recipe().environment,
            "NCCL_MNNVL_ENABLE": "0",
        }
    )
