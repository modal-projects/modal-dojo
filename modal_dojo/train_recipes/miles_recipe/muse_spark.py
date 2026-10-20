"""64K Muse Spark B300 recipes derived from radixark/miles PR #4111.

Trains the text model with frozen omni towers and FP32 master weights; Adam
moments use BF16 on disk.
"""

from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import ModelConfig, Muse_Spark_Open
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

# The image bundles Miles 002ec9cf35e2, SGLang bc1c2ac7dfc5, and Megatron
# 7cc41d59347e. Keep those sources together instead of overlaying older commits.
# Miles PR #4125 supplies Torch 2.14 wheels, replacing our TE/FA2/Apex rebuilds:
# https://github.com/radixark/miles/pull/4125
_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in (
            "patch_muse_spark_te_attention",
            "patch_muse_spark_hf_cache",
        )
    ]


def _mismatch_metrics(**kwargs):
    """Report behavior/current-policy ratios without reweighting the PPO loss."""
    from miles.backends.training_utils.loss.hub.corrections import (
        vanilla_tis_function,
    )

    _, _, metrics = vanilla_tis_function(**kwargs)
    return kwargs["pg_loss"], kwargs["loss_masks"], metrics


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
        "88c66b1414aff0f2cd018923b70f7c463391980236171683bdf22f34796b6bf2"
    )
    image_run_commands: list[str] = field(default_factory=_image_patches)
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
    # Use FA2 for training while keeping SGLang's FA4 available for inference.
    # See patch_muse_spark_te_attention for the pinned dependency incompatibility.
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
    # The bundled SGLang keeps TRT-LLM permutation caches on the CPU so BF16
    # expert updates survive colocated pause/resume on B300.
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


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Muse_Spark_Open_64k_Async_Recipe(Muse_Spark_Open_64k_Recipe):
    """Fully async GRPO on 32 training + 8 rollout B300 GPUs.

    Based on Axiombio's separate trainer/rollout layout. Keeps the synchronous
    recipe's checkpoint cache and streamed BF16 Adam moments. Defaults to one
    proof step; this topology requires its own smoke validation.
    """

    async_mode: bool = True
    fully_async: bool = True
    colocate: bool = False
    rollout_num_gpus: int | None = 8

    # Dedicated GPUs retain both models between updates; only optimizer state
    # streams to disk. Miles rejects target=disk when actor offload is disabled.
    offload_train: bool = False
    offload_rollout: bool = False
    no_offload_train: bool = True
    no_offload_rollout: bool = True
    offload_train_target: str = "cpu"

    # One batch can generate while another trains. Bound finished work to one
    # batch and reject groups whose oldest generated token is >2 versions old.
    async_max_concurrent_samples: int = 16
    async_data_buffer_capacity_factor: float = 1.0
    max_weight_staleness: int = 2
    update_weights_interval: int = 1
    pause_generation_mode: str = "retract"
    # Anchor PPO ratios to the probabilities recorded during generation, even
    # when a response spans weight publications. Keep the actor scoring pass
    # for train/rollout agreement metrics during validation.
    use_rollout_logprobs: bool = True
    get_mismatch_metrics: bool = True
    custom_tis_function_path: str = (
        "modal_dojo.train_recipes.miles_recipe.muse_spark._mismatch_metrics"
    )

    sglang_max_running_requests: int | None = 16
    sglang_server_concurrency: int | None = 16
    sglang_cuda_graph_bs_decode: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 12, 16]
    )
