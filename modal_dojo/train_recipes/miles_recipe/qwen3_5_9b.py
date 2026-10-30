from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_5_9B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_9B_Recipe(MilesRecipe):
    """Qwen3.5-9B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_5_9B
    sft_supported: ClassVar[bool] = False

    miles_model_name: str = "qwen3.5-9B"
    recompute_granularity: str | None = "full"

    colocate: bool = False

    async_mode: bool = True

    actor_num_gpus_per_node: int = 2

    rollout_num_gpus: int | None = 2

    rollout_num_gpus_per_engine: int = 2

    tensor_model_parallel_size: int = 2

    rollout_batch_size: int = 32

    global_batch_size: int = 64

    use_distributed_optimizer: bool = True

    optimizer_cpu_offload: bool = True

    overlap_cpu_optimizer_d2h_h2d: bool = True

    use_precision_aware_optimizer: bool = True

    sglang_mem_fraction_static: float = 0.6

    attention_backend: str = "flash"

    max_tokens_per_gpu: MicroBatchSize = 12288

    lr: float = 5e-7
