from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import Kimi_K3, ModelConfig
from modal_dojo.common.patches import encode_patch
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
    "patch_k3_h200_lora_sync_stream_pp",
    "patch_ipc_bucket_empty_cache",
    "patch_checkpoint_local_dirs",
    "patch_lora_initial_offload",
    "patch_k3_marlin_padding",
    "patch_k3_lora_cpu_staging",
    "patch_k3_lora_health",
    "patch_k3_checkpoint_cpu_merge",
    "patch_k3_fused_activation",
    "patch_k3_fused_lora",
    "patch_k3_fp8_checkpoint",
    "patch_k3_h200_backward_cache",
    "patch_k3_h200_checkpoint_offload",
    "patch_k3_h200_startup_timing",
    "patch_k3_h200_lora_offload_export",
)
_DOCKER_IMAGE = "radixark/miles:dev-202609251434"
_KERNEL_CACHE_ROOT = f"/checkpoints/.kernel-cache/{_DOCKER_IMAGE.split(':')[-1]}"


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in _PATCHES
    ]


def remaining_response_tokens(
    context_length: int, current_length: int, response_limit: int
) -> int:
    if context_length < 2 or current_length < 0 or response_limit < 0:
        raise ValueError("Invalid context, current sequence length, or response limit")
    return max(0, min(response_limit, context_length - current_length - 1))


async def generate_with_context_limit(input):
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.sglang_rollout import generate
    from miles.utils.types import Sample

    sample = input.sample
    context_length = input.args.sglang_context_length
    prompt_length = len(
        input.state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    )
    if prompt_length >= context_length - 1:
        raise ValueError("Prompt leaves no generation space in the K3 context window")
    params = dict(input.sampling_params)
    params["max_new_tokens"] = remaining_response_tokens(
        context_length, prompt_length, params["max_new_tokens"]
    )
    if (
        sample.response
        and len(sample.tokens) - prompt_length >= params["max_new_tokens"]
    ):
        sample.status = Sample.Status.TRUNCATED
        return GenerateFnOutput(samples=sample)

    sample = await generate(input.args, sample, params, evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)


def _extra_config() -> dict:
    return {
        "rollout_max_prompt_len": None,
        "sglang_context_length": 65536,
        "sglang_chunked_prefill_size": 4096,
        "seq_length": 65536,
        "max_position_embeddings": 65536,
        "log_probs_max_tokens_per_gpu": 16384,
        "fp8": "hybrid",
        "fp8_recipe": "delayed",
        "fp8_param_gather": True,
    }


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Kimi_K3_H200_LoRA_Recipe(MilesRecipe):
    name: str = "miles-kimi_k3_lora_recipe"
    custom_generate_function: Callable | None = generate_with_context_limit
    rollout_max_response_len: int = 65536
    eval_max_response_len: int = 65536
    sglang_mem_fraction_static: float = 0.95
    extra_config: dict | None = field(default_factory=_extra_config)
    train_env_vars: dict | str | None = field(
        default_factory=lambda: {
            "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,roundup_power2_divisions:[512:0,>:8],max_split_size_mb:512,per_process_memory_fraction:0.92"
        }
    )

    model_config_class: ClassVar[type[ModelConfig]] = Kimi_K3
    docker_image: str = _DOCKER_IMAGE
    image_run_commands: list[str] = field(default_factory=_image_patches)
    gpu_type: str = "H200"
    memory: tuple[int, int] = (1792 * 1024, 1920 * 1024)
    miles_model_name: str = "kimi-k3"
    model_name: str = "kimi_k3"
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "HF_MODULES_CACHE": "/tmp/hf_modules",
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_RAS_ENABLE": "0",
            "GLOO_SOCKET_IFNAME": "eth1",
            "TP_SOCKET_IFNAME": "eth1",
            "NCCL_TIMEOUT": "3600",
            "CONVERT_DEQUANT_MXFP4": "1",
            "TRITON_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/torchinductor",
            "TILELANG_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/tilelang",
            "SGLANG_CACHE_DIR": f"{_KERNEL_CACHE_ROOT}/sglang",
            "SGLANG_JIT_ROUTE_RADIX": "1",
            "SGLANG_NUMA_BIND_V2": "0",
            "SGLANG_K3_AR_FUSION": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8",
        }
    )
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/Kimi-K3_H200_tp2_pp8_ep8_12_9_torch_dist"
    conversion_tensor_model_parallel_size: int = 2
    conversion_pipeline_model_parallel_size: int = 8
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    convert_ephemeral_disk_mb: int | None = 2 * 1024 * 1024
    download_timeout_seconds: int | None = 8 * 60 * 60
    convert_timeout_seconds: int | None = 12 * 60 * 60
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 8
    decoder_first_pipeline_num_layers: int | None = 12
    decoder_last_pipeline_num_layers: int | None = 9
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 8
    lora_rank: int | None = 32
    lora_alpha: int | None = 64
    target_modules: str | None = "all-linear"
    experts_shared_outer_loras: bool = True
    lora_base_cpu_backup: bool = True
    no_gradient_accumulation_fusion: bool = True
    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 4
    global_batch_size: int = 4
    use_dynamic_global_batch_size: bool = True
    balance_data: bool = True
    use_miles_router: bool = True
    miles_router_max_connections: int = 68
    skip_eval_before_train: bool = True
    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    max_tokens_per_gpu: int = 16384
    log_probs_chunk_size: int = 512
    distributed_timeout_minutes: int = 60
    lr: float = 1e-05
    use_distributed_optimizer: bool = True
    optimizer_cpu_offload: bool = True
    optimizer_offload_fraction: float = 1.0
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    offload_train: bool = True
    colocate_memory_peak_device: str = "cpu"
    update_weight_buffer_size: int | None = 256 * 1024**2
    train_memory_margin_bytes: int = 4 * 1024**3
    check_weight_update_skip_list: list[str] = field(
        default_factory=lambda: ["vision_tower.", "mm_projector."]
    )
    check_weight_update_allow_quant_error: bool = True
    rollout_num_gpus_per_engine: int = 16
    sglang_server_concurrency: int | None = 1
    sglang_max_running_requests: int | None = 1
    sglang_max_mamba_cache_size: int = 5
    sglang_max_total_tokens: int = 131072
    sglang_moe_runner_backend: str | None = "marlin"
    sglang_lora_backend: str | None = "triton"
    sglang_lora_strict_loading: bool = True
    sglang_max_lora_rank: int = 32
    sglang_max_loras_per_batch: int = 1
    no_sglang_lora_use_virtual_experts: bool = True
    sglang_lora_no_cpu_backup: bool = True
    sglang_decode_attention_backend: str | None = "flashinfer"
    sglang_mamba_radix_cache_strategy: str | None = "extra_buffer"
    sglang_cuda_graph_bs_decode: list[int] | None = field(default_factory=lambda: [1])
    sglang_cuda_graph_backend_prefill: str | None = "disabled"
    sglang_weight_loader_prefetch_checkpoints: bool = True
    sglang_weight_loader_prefetch_num_threads: int = 1
    rollout_health_check_first_wait: int = 7200

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "Kimi_K3_H200_LoRA_Recipe":
        patches = _image_patches()
        current = list(self.image_run_commands or [])
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self

    @model_validator(mode="after")
    def _keep_context_settings(self) -> "Kimi_K3_H200_LoRA_Recipe":
        self.extra_config = {**_extra_config(), **(self.extra_config or {})}
        self.environment.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8"
        )
        if (
            self.gpu_type != "H200"
            or self.context_parallel_size != 4
            or self.tensor_model_parallel_size != 2
        ):
            raise ValueError("The H200 recipe requires H200 GPUs and TP2/CP4")
        return self
