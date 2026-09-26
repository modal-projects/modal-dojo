from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_27B_Recipe(SlimeRecipe):
    """Qwen3.6-27B recipe."""

    loss_mask_type: Literal["qwen", "qwen3", "qwen3_5", "distill_qwen"] = "qwen3_5"
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    slime_model_script: str = "scripts/models/qwen3.5-27B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-27B"
    ref_load: str = "/checkpoints/Qwen3.6-27B_torch_dist_tp1pp1"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    gpu_type: str = "B300"
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    conversion_tensor_model_parallel_size: int | None = None
    conversion_pipeline_model_parallel_size: int | None = None
    decoder_last_pipeline_num_layers: int | None = None

    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None
    sglang_mamba_scheduler_strategy: str | None = "extra_buffer"
    sglang_server_concurrency: int | None = None

    max_tokens_per_gpu: int = 8192
    log_probs_chunk_size: int | None = None
    save_debug_rollout_data: str | None = None
    calculate_per_token_loss: bool = True
    balance_data: bool = True
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    attention_backend: str = "unfused"
    eps_clip_high: float | None = None
