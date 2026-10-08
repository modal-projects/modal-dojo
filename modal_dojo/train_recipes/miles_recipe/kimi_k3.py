"""Kimi-K3 LoRA GRPO recipe, ported from upstream's ``run_kimi_k3.py``."""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import Kimi_K3, ModelConfig
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

_PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "frameworks"
    / "miles"
    / "modal_helpers"
    / "patches"
)

_PATCHES = (
    "patch_cell_tick_timeout",
    "patch_lora_sync_stream_pp",
    "patch_ipc_bucket_empty_cache",
    "patch_checkpoint_local_dirs",
    "patch_lora_initial_offload",
)

# Includes K3 support in Miles, Megatron, and SGLang.
_DOCKER_IMAGE = "radixark/miles:dev-202609251434"

# Reuse compiled kernels only within the same image version.
_KERNEL_CACHE_ROOT = f"/checkpoints/.kernel-cache/{_DOCKER_IMAGE.split(':')[-1]}"


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


# KDA's extra-buffer strategy needs five cache slots per running request.
_ROLLOUT_MAX_CONCURRENCY = 8


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_LoRA_Recipe(MilesRecipe):
    """Kimi-K3 rank-32 LoRA recipe for 8 nodes with 8 B300 GPUs each."""

    model_config_class: ClassVar[type[ModelConfig]] = Kimi_K3

    docker_image: str = _DOCKER_IMAGE
    image_run_commands: list[str] = field(default_factory=_image_patches)
    gpu_type: str = "B300"
    # Host backups hold frozen weights for eight ranks per node.
    memory: tuple[int, int] = (1792 * 1024, 2048 * 1024)

    # Upstream's model script supplies the KDA/MLA architecture and layer spec.
    miles_model_name: str = "kimi-k3"
    # Selects Miles' Megatron-to-HF weight mapping.
    model_name: str = "kimi_k3"

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            # Container-local remote code, warmed once per node by the launcher.
            "HF_MODULES_CACHE": "/tmp/hf_modules",
            # Multi-node B300 on Modal has no MNNVL fabric.
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
            # Cross-node Gloo/TCP must use eth1, not the hostname's loopback.
            "GLOO_SOCKET_IFNAME": "eth1",
            "TP_SOCKET_IFNAME": "eth1",
            "NCCL_TIMEOUT": "3600",
            # Dequantize on read to avoid an intermediate BF16 HF checkpoint.
            "CONVERT_DEQUANT_MXFP4": "1",
            "TRITON_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/torchinductor",
            "TILELANG_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/tilelang",
            "SGLANG_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/sglang",
            "SGLANG_JIT_ROUTE_RADIX": "1",
            # sglang's membind pins the whole host backup to one NUMA node,
            # which cannot hold it.
            "SGLANG_NUMA_BIND_V2": "0",
            # The LoRA wrapper bypasses the o_proj.forward patch the K3
            # all-reduce fusion relies on.
            "SGLANG_K3_AR_FUSION": "0",
        }
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Kimi-K3_torch_dist"
    # EP64 spreads conversion's host-memory load across all eight nodes.
    # torch_dist reshards to the training layout on load.
    conversion_tensor_model_parallel_size: int = 32
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 64
    conversion_expert_tensor_parallel_size: int = 1
    # Local staging for each node's share of the ~5.6 TB BF16 checkpoint.
    convert_ephemeral_disk_mb: int | None = 2 * 1024 * 1024
    # 1.5 TB in, ~5.6 TB out: neither fits the launcher's 4-hour stage default.
    download_timeout_seconds: int | None = 8 * 60 * 60
    convert_timeout_seconds: int | None = 12 * 60 * 60

    # ── Cluster ──────────────────────────────────────────────────────────────
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8

    # ── Parallelism: TP4 x CP2 x PP8 = 64 (DP1), EP8 inside each stage ───────
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 8
    # 93 = 12 x 7 + 9 across 8 stages.
    decoder_first_pipeline_num_layers: int | None = 12
    decoder_last_pipeline_num_layers: int | None = 9
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 8

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_rank: int | None = 32
    lora_alpha: int | None = 64
    # Resolve K3's HF targets; this image rejects the older Megatron names.
    target_modules: str | None = "all-linear"
    # One A factor shared across the 896 routed experts, per-expert B factors.
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True

    # ── Rollout and sampling ─────────────────────────────────────────────────
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    use_dynamic_global_batch_size: bool = True
    balance_data: bool = True
    use_miles_router: bool = True
    miles_router_max_connections: int = 68
    skip_eval_before_train: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    max_tokens_per_gpu: MicroBatchSize = 8192
    log_probs_chunk_size: int = 512
    distributed_timeout_minutes: int = 60

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr: float = 1e-5
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    optimizer_offload_fraction: float = 0.8
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Colocation and weight sync ───────────────────────────────────────────
    offload_train: bool = True
    # Overlap trainer/engine residency on GPU to reduce peak host memory.
    colocate_memory_peak_device: str = "gpu"
    # Only the adapter is transferred each step.
    update_weight_buffer_size: int | None = 256 * 1024**2
    train_memory_margin_bytes: int = 4 * 1024**3
    # The trainer omits vision weights; base weights differ by quantization.
    check_weight_update_skip_list: list[str] = field(
        default_factory=lambda: ["vision_tower.", "mm_projector."]
    )
    check_weight_update_allow_quant_error: bool = True

    # ── SGLang: one TP16 engine spans two nodes, experts replicated over TP ──
    rollout_num_gpus_per_engine: int = 16
    sglang_server_concurrency: int | None = 16
    sglang_max_running_requests: int | None = _ROLLOUT_MAX_CONCURRENCY
    sglang_max_mamba_cache_size: int = 5 * _ROLLOUT_MAX_CONCURRENCY
    sglang_max_total_tokens: int = 65536
    # Marlin supports MXFP4 experts with LoRA.
    sglang_moe_runner_backend: str | None = "marlin"
    sglang_lora_backend: str | None = "triton"
    sglang_lora_strict_loading: bool = True
    sglang_max_lora_rank: int = 32
    sglang_max_loras_per_batch: int = 1
    # Above TP8 the Marlin MoE intermediate is tile-padded, which the
    # virtual-experts LoRA kernel rejects.
    no_sglang_lora_use_virtual_experts: bool = True
    # Adapters are streamed each step, so their host backup is unused.
    sglang_lora_no_cpu_backup: bool = True
    sglang_decode_attention_backend: str | None = "trtllm_mla"
    sglang_mamba_radix_cache_strategy: str | None = "extra_buffer"
    sglang_cuda_graph_bs_decode: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8]
    )
    sglang_cuda_graph_backend_prefill: str | None = "disabled"
    # Each engine loads and repacks the full 1.5 TB release before health checks.
    rollout_health_check_first_wait: int = 7200

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_LoRA_Recipe":
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
