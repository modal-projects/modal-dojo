"""What every merged-LoRA recipe on Stitch shares: a frozen BF16 base trained on B300s, rollout on Stitch engines.

Training is fully async. Each version's merged adapter reaches the engines as a weight delta, with checksums.
Score centering with masked importance sampling learns from the sampler's own top-k candidates, which corrects
for the samples' staleness. The engines serve a converted copy of the base (``hf_checkpoint``) that the trainer's
export matches; each model's recipe says how to make it.
"""

from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

# The Miles image, and the Miles and SGLang commits Stitch's weight delta runs on.
STITCH_MILES_IMAGE = "radixark/miles:dev-202611121556"
STITCH_MILES_REF = "c2d56413f5683ed72fa815067abcf0c25b413858"
STITCH_SGLANG_REF = "69d57a05097d869ebcaee3f5fbe7338b740b282b"

# MilesRecipe.environment's default.
MILES_ENVIRONMENT = {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
}


def stitch_lora_extra_config(**model: Any) -> dict[str, Any]:
    """The transfer and loss settings every Stitch LoRA recipe uses, with the model's own on top."""
    return {
        "fully_async": True,
        "update_weight_transfer_mode": "weight-delta",
        "check_weight_transfer_checksum": True,
        "lora_merge_for_rollout": True,
        "loss_type": "score_centering",
        "score_centering_is": "mis",
        "rollout_top_logprobs_num": 128,
        "use_rollout_logprobs": True,
        "grpo_std_normalization": False,
        "recompute_loss_function": True,
        "seq_length": 32768,
        "max_seq_len": 32768,
        "max_position_embeddings": 32768,
        "bf16": True,
        "sglang_moe_a2a_backend": "none",
        **model,
    }


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _Stitch_LoRA_Recipe(MilesRecipe):
    gpu_type: str = "B300"
    colocate: bool = False
    stitch: bool = True
    async_mode: bool = True
    actor_num_gpus_per_node: int = 8
    docker_image: str = STITCH_MILES_IMAGE
    miles_git_ref: str | None = STITCH_MILES_REF
    sglang_git_ref: str | None = STITCH_SGLANG_REF
    megatron_to_hf_mode: str = "raw"

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    balance_data: bool = True
    no_gradient_accumulation_fusion: bool = True

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    target_modules: str | None = "all-linear"
    experts_shared_outer_loras: bool = True

    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    rollout_max_response_len: int = 16384
    # Score centering's candidates: the sampler's top-k under top-p.
    rollout_top_p: float = 0.97
    rollout_top_k: int | None = 128
    calculate_per_token_loss: bool = True

    lr: float = 1e-5
    weight_decay: float = 0.0

    sglang_enable_dp_lm_head: bool = True
    extra_config: dict | None = field(default_factory=stitch_lora_extra_config)
