"""Inkling 2 Small LoRA recipe on one B300 node, mirroring upstream Miles.

The BF16 trainer and the engine serving the released NVFP4/MXFP8 experts do not
fit on the GPUs together, so the trainer is parked on local disk during rollouts
and only the adapter is synced.
"""

from dataclasses import field
from typing import Any, ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import Inkling_2_Small, ModelConfig
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Small_LoRA_Recipe(MilesRecipe):
    """Rank-32 LoRA GRPO, 8,192-token context, 1 node × 8 B300 GPUs.

    Defaults to one proof step; increasing the horizon is explicit.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Inkling_2_Small
    # Miles radixark/miles#4819 with an SGLang build that serves the model.
    # Its Miles and SGLang trees are not git checkouts, so the git-ref overlays
    # cannot be used on it.
    docker_image: str = (
        "radixark/miles:inkling-2-small@sha256:"
        "a2e3c49d5a266d9374d67ca4b706767b7bdf7d798958dd51520f61c29597e9a5"
    )
    miles_model_name: str = "inkling2"

    gpu_type: str = "B300"
    actor_num_gpus_per_node: int = 8
    num_gpus_per_node: int = 8
    # The engine's CPU backup of the quantized base and the page cache of the
    # disk-offloaded trainer both count toward host memory.
    memory: int | tuple[int, int] | None = (2560 * 1024, 3072 * 1024)
    cpu: float | tuple[float, float] | None = 32.0
    expert_model_parallel_size: int = 8

    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Inkling-2-Small_torch_dist"
    # The dequantized experts do not fit one GPU at EP1.
    conversion_tensor_model_parallel_size: int = 1
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    # The 1.1 TiB torch_dist is staged locally, and its uncommitted Volume copy
    # occupies the same disk.
    convert_ephemeral_disk_mb: int | None = 3 * 1024 * 1024
    download_timeout_seconds: int | None = 8 * 60 * 60
    convert_timeout_seconds: int | None = 8 * 60 * 60
    # The offloaded trainer is about 1.3 TiB on node-local disk.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": 3 * 1024 * 1024}
    )
    max_retries: int = 0

    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    balance_data: bool = True
    rm_type: str | None = "math"
    # Corrects for the gap between the BF16 trainer and the quantized engine.
    use_tis: bool = True
    use_rollout_routing_replay: bool = True
    lr: float = 5e-5
    use_distributed_optimizer: bool = True

    bf16: bool = True
    attention_softmax_in_fp32: bool = False
    no_bias_dropout_fusion: bool = True
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    max_tokens_per_gpu: MicroBatchSize = 8192
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/tmp/miles_train_offload"

    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    target_modules: str | None = "all-linear"
    experts_shared_outer_loras: bool = True
    # A resynced NVFP4 base would come back requantized, so the engine keeps
    # the base it loaded and syncs carry the adapter only.
    lora_base_cpu_backup: bool = True
    check_lora_weight_equal: bool = True
    sglang_lora_backend: str | None = "triton"
    sglang_max_loras_per_batch: int = 1
    sglang_max_lora_rank: int = 32

    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int | None = 8
    sglang_attention_backend: str | None = "fa4"
    sglang_bf16_gemm_backend: str = "torch"
    sglang_page_size: int = 128
    sglang_mem_fraction_static: float = 0.6
    sglang_context_length: int = 8192
    environment: dict[str, str] = field(
        default_factory=lambda: {
            # The inkling2 plugin lives in the Miles checkout and is not installed.
            "PYTHONPATH": "/root/miles:/root/Megatron-LM",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "0",
            # The image's SGLang is a source build ahead of its sgl-kernel wheel.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # Publishing 1.1 TiB to the Volume inside the save outlasts NCCL's
            # ten-minute watchdog.
            "CONVERT_LOCAL_STAGING": "1",
        }
    )
