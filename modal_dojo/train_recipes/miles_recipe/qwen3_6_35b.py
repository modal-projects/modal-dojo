from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Qwen3_6_35B
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

from modal_dojo.train_recipes.base import MicroBatchSize


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35B_Recipe(MilesRecipe):
    """Qwen3.6-35B-A3B recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_6_35B
    sft_supported: ClassVar[bool] = False

    miles_model_name: str = "qwen3.6-35B-A3B"
    recompute_granularity: str | None = "full"

    gpu_type: str = "B300"

    colocate: bool = False

    async_mode: bool = True

    actor_num_gpus_per_node: int = 4

    rollout_num_gpus: int | None = 4

    tensor_model_parallel_size: int = 2

    sequence_parallel: bool = True

    use_distributed_optimizer: bool = True

    rollout_batch_size: int = 32

    n_samples_per_prompt: int = 8

    global_batch_size: int = 256

    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"

    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    expert_model_parallel_size: int = 2

    sglang_ep_size: int | None = 1

    sglang_attention_backend: str | None = "triton"

    sglang_moe_runner_backend: str | None = "flashinfer_trtllm_routed"

    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )

    sglang_max_running_requests: int | None = 512

    max_tokens_per_gpu: MicroBatchSize = 16384

    balance_data: bool = True

    optimizer_cpu_offload: bool = True

    overlap_cpu_optimizer_d2h_h2d: bool = True

    use_precision_aware_optimizer: bool = True

    memory: int | tuple[int, int] | None = (128, 2_097_152)

    attention_backend: str = "unfused"

    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp1"

    no_save_optim: bool = True
