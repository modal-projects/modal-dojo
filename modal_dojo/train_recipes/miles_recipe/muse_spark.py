"""64K Muse Spark B300 recipe derived from radixark/miles PR #4111.

Validated with a five-step SWE-bench smoke on 32 B300s. Trains the text model
with frozen omni towers and FP32 master weights; Adam moments use BF16 on disk.
"""

from collections.abc import Callable
from dataclasses import field
from typing import Any, ClassVar

import modal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Muse_Spark_Open
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_dojo.frameworks.miles.modal_helpers.patches.patch_muse_spark_native import (
    rebuild_te,
    rebuild_fa2,
    rebuild_apex,
    patch_te_fa4_gate,
)
from modal_dojo.frameworks.miles.modal_helpers.patches.patch_muse_spark_hf_cache import (
    patch_muse_hf_cache_shards,
)

MILES_COMMIT = "753e19d8bd8fe5a27892dd18b787dc131addd269"
SGLANG_COMMIT = "ad34947129c518641db51816c88c126ca16f786c"


def build_muse_native_image(image: modal.Image) -> modal.Image:
    """Build and cache native wheels separately before starting GPU workers."""
    return (
        image.run_function(rebuild_te, cpu=16, memory=65536, timeout=3600)
        .run_function(rebuild_fa2, cpu=16, memory=98304, timeout=7200)
        .run_function(rebuild_apex, cpu=16, memory=65536, timeout=3600)
        .run_function(patch_te_fa4_gate, timeout=60)
        .run_function(patch_muse_hf_cache_shards, timeout=60)
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Muse_Spark_Open_64k_Recipe(MilesRecipe):
    """Full-parameter GRPO, 65,536-token context, 4 nodes × 8 B300 GPUs.

    Uses upstream's TP8/PP4/CP1/EP8 layout to keep expert communication within
    each node. Defaults to one proof step; increasing the horizon is explicit.
    Node-local disk holds streamed Adam state; optimizer state is not saved.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Muse_Spark_Open
    # Preserve the volume name holding the validated torch_dist conversion.
    name: str = "miles-muse_spark_open_64k_b300_recipe"
    docker_image: str = (
        "radixark/miles:muse-spark-open@sha256:"
        "2b46ca326003e347c600145bd60935b42af8032ed6f0a75e7fdc7bf5a8700b15"
    )
    image_overlay: Callable | None = field(
        default_factory=lambda: build_muse_native_image
    )
    miles_git_ref: str = MILES_COMMIT
    sglang_git_ref: str = SGLANG_COMMIT
    miles_model_name: str = "muse-spark-full"
    model_name: str = "muse_spark"

    gpu_type: str = "B300"
    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8
    # Disk cache counts toward host memory despite not appearing in guest RSS.
    # A 768 GiB reservation was evicted on step two; 2.5 TiB passed five steps.
    memory: int | tuple[int, int] | None = (2560 * 1024, 3072 * 1024)
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 1
    cp_comm_type: str = ""
    expert_model_parallel_size: int = 8
    # TE's FA4 backward is incompatible with pinned quack 0.6.5. Use rebuilt
    # FA2 for training while keeping SGLang's FA4 available for inference.
    train_env_vars: dict[str, str] | str | None = field(
        default_factory=lambda: {"DOJO_MUSE_TE_FA4": "0"}
    )
    decoder_first_pipeline_num_layers: int = 15
    decoder_last_pipeline_num_layers: int | None = 15
    sequence_parallel: bool = True

    # Convert on four nodes with upstream's TP1/PP4/EP8 layout; torch_dist
    # reshards to the trainer's TP8 layout on load.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Muse-Spark-Open-816B-A42B_torch_dist"
    conversion_tensor_model_parallel_size: int = 1
    conversion_pipeline_model_parallel_size: int = 4
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    convert_ephemeral_disk_mb: int | None = 1024 * 1024
    download_timeout_seconds: int | None = 12 * 60 * 60
    convert_timeout_seconds: int | None = 12 * 60 * 60
    # Leave room for streamed Adam state and BF16 actor backups on each node.
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": 3 * 1024 * 1024}
    )
    no_save_optim: bool = True
    no_load_optim: bool = True
    save: str | None = None
    max_retries: int = 0
    # Resharding the 1.6 TB checkpoint can leave ranks >10 minutes apart at
    # Megatron's final load barrier.
    distributed_timeout_minutes: int = 30

    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    global_batch_size: int = 16
    num_steps_per_rollout: int = 1
    rollout_max_context_len: int = 65536
    rollout_max_prompt_len: int = 4096
    rollout_max_response_len: int = 61440
    eval_max_response_len: int = 61440
    seq_length: int = 65536
    max_position_embeddings: int = 65536
    apply_chat_template_kwargs: str | dict = field(
        default_factory=lambda: {"reasoning_effort": "xhigh"}
    )
    chat_template_path: str = (
        "/root/miles/miles/utils/chat_template_utils/templates/muse_spark_fixed.jinja"
    )
    # Leave reward to the workload.
    rm_type: str = ""
    use_rollout_routing_replay: bool = True
    calculate_per_token_loss: bool = True
    balance_data: bool = True
    # Packing budget; a single long sample may exceed it, as in upstream.
    max_tokens_per_gpu: int = 8192
    log_probs_chunk_size: int = 4096
    attention_backend: str = "flash"
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    bf16: bool = True
    min_lr: float = 1e-6
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    clip_grad: float = 0.0
    use_distributed_optimizer: bool = True
    ckpt_format: str = "torch_dist"
    stream_optimizer_state_to_disk: bool = True
    stream_optimizer_state_moment_dtype: str = "bf16"
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/root/muse_spark_optimizer"
    offload_train_disk_chunk_mb: int = 256

    # Four single-node engines, with BF16 KV for train/rollout agreement.
    # The SGLang pin includes 358c17d4b4f5, required for BF16 TRT-LLM expert
    # updates after colocated pause/resume on B300.
    offload_train: bool = True
    offload_rollout: bool = True
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int | None = 8
    sglang_pp_size: int = 1
    sglang_dtype: str = "bfloat16"
    sglang_kv_cache_dtype: str = "bfloat16"
    sglang_mem_fraction_static: float = 0.83
    sglang_context_length: int = 65536
    sglang_chunked_prefill_size: int = 1024
    sglang_max_running_requests: int | None = 4
    sglang_server_concurrency: int | None = 4
    sglang_cuda_graph_bs_decode: list[int] = field(default_factory=lambda: [1, 2, 4])
    sglang_cuda_graph_backend_prefill: str = "disabled"
    sglang_router_policy: str = "round_robin"
    update_weight_buffer_size: int = 2 * 1024**3
    pin_rollout_manager_to_head: bool = True
    rollout_health_check_first_wait: int = 3600
    skip_eval_before_train: bool = True
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # Publishing ~1.6 TB must not wait on a GPU collective's 10m timer.
            "MODAL_DOJO_CHECKPOINT_COMMIT_TIMEOUT_SECONDS": "3600",
            # Matches upstream B300 validation; CUTLASS 4.6.2 lacks experimental.
            "SGLANG_ENABLE_BF16_SPLITK_GEMM": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }
    )
