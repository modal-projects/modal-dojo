from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35B_Recipe(SlimeRecipe):
    """Qwen3.6-35B-A3B recipe."""

    loss_mask_type: Literal["qwen", "qwen3", "qwen3_5", "distill_qwen"] = "qwen3_5"
    gpu_type: str = "B300"
    colocate: bool = False
    rollout_num_gpus: int | None = 1
    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sglang_ep_size: int | None = 1
    sglang_attention_backend: str | None = "triton"
    sglang_moe_runner_backend: str | None = "flashinfer_trtllm_routed"
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )
    sglang_max_running_requests: int | None = 512
    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None
    sglang_mamba_scheduler_strategy: str = "extra_buffer"

    max_tokens_per_gpu: int = 8192
    balance_data: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_enable_deepep: bool = False

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    memory: int | tuple[int, int] | None = (128, 2_097_152)
    attention_backend: str = "unfused"

    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp1pp1"
    no_save_optim: bool = True
