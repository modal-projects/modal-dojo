"""Candidate 64K Muse Spark recipe derived from radixark/miles PR #4111.

Upstream validated 8K, not 64K. This preset needs a one-step GPU proof and
smoke test before production use. It trains the text model, with frozen omni
towers. No LoRA variant is claimed: upstream validated full-parameter RL.
"""

from collections.abc import Callable
from dataclasses import field
from typing import ClassVar

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


def warmup_muse_pipeline(args, model, store_prefix: str) -> None:
    """Initialize the pipeline communicator together after trainer wake-up."""
    import torch
    from megatron.core import parallel_state

    if parallel_state.get_pipeline_model_parallel_world_size() <= 1:
        return
    # Offloading destroys NCCL communicators. A barrier initializes collective
    # channels, but NCCL still connects P2P channels lazily. Connect both
    # directions before a receiver can wait on a sender busy with MoE kernels.
    group = parallel_state.get_pipeline_model_parallel_group()
    torch.distributed.barrier(group=group, device_ids=[torch.cuda.current_device()])
    ranks = torch.distributed.get_process_group_ranks(group)
    stage = torch.distributed.get_rank(group)
    ops = []
    for peer_stage in (stage - 1, stage + 1):
        if 0 <= peer_stage < len(ranks):
            send = torch.zeros(1, device=torch.cuda.current_device())
            recv = torch.empty_like(send)
            ops.extend(
                [
                    torch.distributed.P2POp(
                        torch.distributed.isend, send, ranks[peer_stage], group
                    ),
                    torch.distributed.P2POp(
                        torch.distributed.irecv, recv, ranks[peer_stage], group
                    ),
                ]
            )
    for request in torch.distributed.batch_isend_irecv(ops):
        request.wait()
    # Keep every PP group out of overlapping MoE work until connections finish.
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    torch.cuda.synchronize()
    if torch.distributed.get_rank() == 0:
        print(
            f"[muse] pipeline P2P connections ready before {store_prefix} log-probs",
            flush=True,
        )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Muse_Spark_Open_64k_Recipe(MilesRecipe):
    """Full-parameter GRPO, 65,536-token context, 8 nodes × 8 H200 GPUs.

    Uses the upstream-supported TP4/PP4/CP4/EP16 layout to shard long
    sequences. Defaults to one proof step; increasing the horizon is explicit.
    Node-local disk holds streamed Adam state; optimizer state is not saved.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Muse_Spark_Open
    # Both hardware variants reuse the same completed torch_dist conversion.
    # Complete preparation for one launch before submitting the other.
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

    gpu_type: str = "H200"
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8
    memory: int | tuple[int, int] | None = (512 * 1024, 1792 * 1024)
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 4
    cp_comm_type: str = "a2a"
    expert_model_parallel_size: int = 16
    # Load trainer CUDA code before MoE collectives can overlap the first
    # cuBLAS grouped GEMM. Lazy kernel initialization can deadlock that overlap.
    train_env_vars: dict | str | None = field(
        default_factory=lambda: {
            "CUDA_MODULE_LOADING": "EAGER",
            "CUDA_MODULE_DATA_LOADING": "EAGER",
        }
    )
    decoder_first_pipeline_num_layers: int = 15
    decoder_last_pipeline_num_layers: int | None = 15
    sequence_parallel: bool = True

    # Convert on four nodes with the upstream B300 TP1/PP4/EP8 layout.
    # The pinned tensor headers give <=57 GiB of text weights per rank,
    # leaving H200 headroom without requiring all eight training nodes.
    # torch_dist reshards to the H200 trainer's EP16 layout on load.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Muse-Spark-Open-816B-A42B_torch_dist"
    conversion_tensor_model_parallel_size: int = 1
    conversion_pipeline_model_parallel_size: int = 4
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    convert_ephemeral_disk_mb: int | None = 1024 * 1024
    download_timeout_seconds: int | None = 12 * 60 * 60
    convert_timeout_seconds: int | None = 12 * 60 * 60
    # ~1.2 TB/node optimizer storage upstream, plus weight backups/staging.
    train_function_kwargs: dict = field(
        default_factory=lambda: {"ephemeral_disk": 2048 * 1024}
    )
    no_save_optim: bool = True
    no_load_optim: bool = True
    save: str | None = None
    max_retries: int = 0
    # Resharding the 1.6 TB checkpoint can leave ranks >10 minutes apart at
    # Megatron's final load barrier, especially across eight H200 nodes.
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
    # Leave reward to the workload (math in the example/validation harness).
    rm_type: str = ""
    use_rollout_routing_replay: bool = True
    calculate_per_token_loss: bool = True
    balance_data: bool = True
    # Packing budget; a single long sample may exceed it, as in upstream.
    max_tokens_per_gpu: int = 4096
    log_probs_chunk_size: int = 4096
    custom_megatron_before_log_prob_hook: Callable | str | None = field(
        default_factory=lambda: warmup_muse_pipeline
    )
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
    stream_optimizer_state_moment_dtype: str = "fp32"
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/root/muse_spark_optimizer"
    offload_train_disk_chunk_mb: int = 256

    # Four two-node engines, with BF16 KV for train/rollout agreement.
    offload_train: bool = True
    offload_rollout: bool = True
    rollout_num_gpus_per_engine: int = 16
    sglang_ep_size: int | None = 16
    sglang_pp_size: int = 1
    sglang_dtype: str = "bfloat16"
    sglang_kv_cache_dtype: str = "bfloat16"
    sglang_mem_fraction_static: float = 0.8
    sglang_context_length: int = 65536
    sglang_chunked_prefill_size: int = 1024
    sglang_max_running_requests: int | None = 4
    sglang_server_concurrency: int | None = 4
    sglang_cuda_graph_bs_decode: list[int] = field(default_factory=lambda: [1, 2, 4])
    sglang_cuda_graph_backend_prefill: str = "disabled"
    sglang_router_policy: str = "round_robin"
    # CUDA IPC flattens each bucket into an additional GPU allocation. A 2 GiB
    # bucket exhausted H200 memory during the initial colocated weight sync.
    update_weight_buffer_size: int = 1024**3
    rollout_health_check_first_wait: int = 3600
    skip_eval_before_train: bool = True
    environment: dict = field(
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
class Muse_Spark_Open_64k_B300_Recipe(Muse_Spark_Open_64k_Recipe):
    """Candidate 65,536-token recipe on 4 nodes × 8 B300 GPUs.

    Keeps the training topology of upstream's validated 8K B300 launcher:
    TP8/PP4/CP1/EP8 and four single-node rollout engines. The 64K extension
    remains unvalidated; limit each engine to four concurrent requests for
    the first proof. The pinned SGLang includes commit 358c17d4b4f5, required
    for BF16 TRT-LLM expert updates after colocated pause/resume on B300.
    """

    # Local TE-only switch avoids FA4 backward with pinned quack 0.6.5.
    # SGLang retains its installed FA4; H200 keeps the default FA3 path.
    train_env_vars: dict | str | None = field(
        default_factory=lambda: {"DOJO_MUSE_TE_FA4": "0"}
    )
    gpu_type: str = "B300"
    # Budget for the sandbox's disk cache as well as live CPU tensors. Guest
    # memory statistics omit the rootfs cache: ~0.5 TiB reported usage still
    # led to second-step evictions with a 768 GiB reservation. Reserve Modal's
    # maximum 2.5 TiB per node for state, actor backups, and live CPU tensors.
    memory: int | tuple[int, int] | None = (2560 * 1024, 3072 * 1024)
    custom_megatron_before_log_prob_hook: Callable | str | None = None
    actor_num_nodes: int = 4
    tensor_model_parallel_size: int = 8
    context_parallel_size: int = 1
    cp_comm_type: str = ""
    expert_model_parallel_size: int = 8
    # Half as many nodes doubles per-node disk demand. Three TiB is Modal's
    # maximum; ~2.6 TiB is estimated for FP32 Adam state + BF16 actor backup.
    # Confirm measured free space before a live run; checkpoints stay disabled.
    train_function_kwargs: dict = field(
        default_factory=lambda: {"ephemeral_disk": 3 * 1024 * 1024}
    )
    max_tokens_per_gpu: int = 8192
    rollout_num_gpus_per_engine: int = 8
    sglang_ep_size: int | None = 8
    sglang_mem_fraction_static: float = 0.83
    update_weight_buffer_size: int = 2 * 1024**3
    pin_rollout_manager_to_head: bool = True
