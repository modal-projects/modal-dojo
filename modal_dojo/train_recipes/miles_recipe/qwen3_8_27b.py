from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_8_27B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8_27B_Recipe(MilesRecipe):
    """Qwen3.8-27B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_8_27B
    sft_supported: ClassVar[bool] = False

    miles_model_name: str = "qwen3.8-27B"
    recompute_granularity: str | None = "full"

    memory: int | tuple[int, int] | None = (128, 2_097_152)

    hf_checkpoint: str = "Qwen/Qwen3.8-27B"

    ref_load: str = "/checkpoints/Qwen3.8-27B_torch_dist_tp2pp1"

    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    gpu_type: str = "B300"
    colocate: bool = False
    async_mode: bool = True
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int | None = 4
    rollout_num_gpus_per_engine: int = 4

    tensor_model_parallel_size: int = 2

    sglang_mem_fraction_static: float = 0.6

    sglang_max_running_requests: int | None = 32

    sglang_tool_call_parser: str | None = "qwen3_coder"

    sglang_reasoning_parser: str | None = "qwen3"

    rollout_batch_size: int = 16

    n_samples_per_prompt: int = 4

    global_batch_size: int = 64

    max_tokens_per_gpu: MicroBatchSize = 8192

    use_dynamic_batch_size: bool = False

    calculate_per_token_loss: bool = True

    balance_data: bool = True

    use_distributed_optimizer: bool = True

    optimizer_cpu_offload: bool = True

    overlap_cpu_optimizer_d2h_h2d: bool = True

    use_precision_aware_optimizer: bool = True

    attention_backend: str = "unfused"

    eps_clip_high: float | None = None
