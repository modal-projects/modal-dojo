"""DeepSeek V4.1 Flash native LoRA, based on Miles #4205 / SGLang #45231."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.deepseek_v41_flash import DeepSeek_V4_1_Flash_LoRA
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash import (
    _PATCH_DIR,
    DeepSeek_V4_1_Flash_Recipe,
)

# TODO: Use a released image once Miles #4205 and SGLang #45231 are included.
MILES_LORA_COMMIT = "6c8e2378cfe3dc5f99633e9719a80586dac86683"
SGLANG_LORA_COMMIT = "34033fe90b90f1bb64e63337268beada57f21f84"


def _lora_image_commands() -> list[str]:
    # The full-weight checksum patches belong to the older full-parameter
    # image. Native LoRA verifies the streamed adapter instead.
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in (
            "patch_deepseek_v41_router_registration",
            "patch_deepseek_v41_fp4_dequant_block",
            "patch_deepseek_v41_vision_topk_capture",
            "patch_deepseek_v41_lora_efficiency",
            "patch_checkpoint_local_dirs",
            "patch_native_lora_checkpoint_commit",
            "patch_sglang_weights_disk_backup",
        )
    ]


def _lora_environment() -> dict[str, str]:
    return {
        **DeepSeek_V4_1_Flash_Recipe().environment,
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
        # Adapter-only synchronization leaves room for the TP-sharded Engram
        # tables on H200; avoid host-memory lookups on every decode step.
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
        # Load cuBLAS kernels before expert collectives can overlap their first
        # initialization (CUDA's lazy-loader can stall in cuKernelSetAttribute).
        "CUDA_MODULE_LOADING": "EAGER",
        # Reserve node CPU capacity without multiplying BLAS threads in each
        # of the eight trainer/serving worker processes.
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR": "/root/sglang_weights_offload",
        "TILELANG_CACHE_DIR": "/tmp/tilelang_cache",
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
    }


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_LoRA_Recipe(DeepSeek_V4_1_Flash_Recipe):
    """Rank-32 attention/shared-expert LoRA on 32 H200 GPUs with a 32K window.

    Covers all five attention projections and the shared expert's gate/up/down
    projections supported by Miles #4205 and SGLang #45231. Routed experts,
    compressor, indexer and Engram remain frozen. Workload and reward are
    supplied by the training config; generation uses ordinary stopping.
    """

    model_config_class: ClassVar[type[DeepSeek_V4_1_Flash_LoRA]] = (
        DeepSeek_V4_1_Flash_LoRA
    )
    docker_image: str = (
        "radixark/miles:dev@sha256:"
        "25869dfe011bda93e9ac01833d0ee8ece9aac34cb8940cf19fa46cef322efd16"
    )
    miles_git_ref: str | None = MILES_LORA_COMMIT
    sglang_git_ref: str | None = SGLANG_LORA_COMMIT
    image_run_commands: list[str] = field(default_factory=_lora_image_commands)
    environment: dict[str, str] = field(default_factory=_lora_environment)
    ref_load: str = (
        "/checkpoints/DeepSeek-V4.1-Flash-1017_1bae0e108899_native_lora_torch_dist"
    )

    # A high ceiling with a tiny reservation does not guarantee enough RAM
    # during the full-model offload/restore transition. Reserve the node budget.
    memory: int | tuple[int, int] | None = (1792 * 1024, 1792 * 1024)
    cpu: float | tuple[float, float] | None = 32.0
    actor_num_nodes: int = 4
    expert_model_parallel_size: int = 32
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Fully qualified names avoid the compressor/indexer/Engram suffix aliases.
    target_modules: str = ",".join(
        [f"layers.*.attn.{name}" for name in ("wq_a", "wkv", "wq_b", "wo_a", "wo_b")]
        + [f"layers.*.ffn.shared_experts.{name}" for name in ("w1", "w3", "w2")]
    )
    exclude_modules: str = "layers.*.ffn.experts.*"
    lora_base_cpu_backup: bool = True
    check_lora_weight_equal: bool = True
    check_weight_update: bool = False
    no_gradient_accumulation_fusion: bool = True
    # Serving's ~900 GiB backup per node lives on local disk, leaving host RAM
    # for the trainer's frozen Engram tables and faster CPU offload/restore.
    offload_train_target: str = "cpu"

    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    global_batch_size: int = 32
    rollout_temperature: float = 1.0
    rollout_max_context_len: int = 32768
    rollout_max_prompt_len: int = 4096
    rollout_max_response_len: int = 28672
    seq_length: int = 32768
    max_tokens_per_gpu: MicroBatchSize = 32768
    lr: float = 1e-4
    weight_decay: float = 0.0
    kl_loss_type: str = "k3"
    calculate_per_token_loss: bool = True

    sglang_context_length: int = 32768
    sglang_lora_backend: str = "triton"
    sglang_max_lora_rank: int = 32
    sglang_max_loras_per_batch: int = 1
    sglang_lora_strict_loading: bool = True
    sglang_disable_overlap_schedule: bool = True
    sglang_disable_radix_cache: bool = False
    # The GPU-resident Engram tables add ~24 GiB per TP8 rank to the
    # ~88 GiB serving weights. Leave ~14 GiB outside the static pool.
    sglang_mem_fraction_static: float = 0.9
    sglang_disable_cuda_graph: bool = False
    sglang_cuda_graph_max_bs_decode: int = 16
    sglang_tool_call_parser: str = "deepseekv41"
    sglang_reasoning_parser: str = "deepseek-v41"
    pause_generation_mode: str = "in_place"
    extra_config: dict | None = field(
        default_factory=lambda: {
            "sglang_device": "cuda",
            # Shared SWE-bench prompt prefixes can send an entire 32-response
            # batch to one cache-aware worker (its default balance threshold
            # is 64). Keep all four engines busy and inside decode graph sizes.
            "sglang_router_policy": "round_robin",
            # Weight offload still occurs before training. Allow the full
            # frozen model's disk backup to finish without killing the engine.
            "sglang_watchdog_timeout": 1800,
            "stream_optimizer_state_to_disk": False,
            "sglang_weight_loader_drop_cache_after_load": True,
            "sglang_model_loader_extra_config": '{"num_threads": 2}',
            "use_rollout_logprobs": True,
            "normalize_advantages": False,
            "grpo_std_normalization": True,
            "rewards_normalization": True,
            "use_tis": False,
            "get_mismatch_metrics": True,
            # New Miles requires an explicit metrics implementation. With
            # use_tis=False this preserves the policy loss and response masks.
            "custom_tis_function_path": (
                "examples.infra_features.train_infer_mismatch_helper.mis."
                "compute_mis_weights_with_cp"
            ),
            # The example helper reads these before its metrics-only early
            # return. They do not reweight losses or reject tokens with TIS off.
            "tis_lower_bound": 0.5,
            "tis_upper_bound": 2.0,
            "rs_lower_bound": 0.5,
            "rs_upper_bound": 2.0,
            "recompute_loss_function": False,
            "use_unbiased_kl": False,
            "log_probs_chunk_size": 128,
            "seed": 1234,
            "rollout_seed": 2718,
            "rollout_top_k": -1,
            "adam_eps": 1e-10,
            # Cold reads of the full converted checkpoint can leave the first
            # node waiting over ten minutes for the remaining nodes.
            "distributed_timeout_minutes": 60,
            # Record runtime provenance at startup without repeatedly probing
            # every repository from all 32 training workers.
            "env_report_interval_seconds": 0,
            # Native adapter manifests are checked by check_lora_weight_equal.
            # Hashing every frozen base tensor after each update adds minutes.
            "log_inference_engine_weight_checksums": False,
        }
    )

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "DeepSeek_V4_1_Flash_LoRA_Recipe":
        """Replace the inherited image-specific patch set for the newer runtime."""
        required = _lora_image_commands()
        current = self.image_run_commands or []
        object.__setattr__(
            self,
            "image_run_commands",
            [*required, *(command for command in current if command not in required)],
        )
        if self.lora_rank <= 0 or self.sglang_max_lora_rank != self.lora_rank:
            raise ValueError(
                "Training and serving must use the same positive LoRA rank"
            )
        return self
