"""Native GLM-5.3-Flash LoRA from radixark/miles#3098, with 32k context."""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.glm_5_3_flash import GLM_5_3_Flash_LoRA
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)


def _image_commands() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in (
            "patch_glm_5_3_flash_lora_checkpoint",
            # Initial serving-weight offload can exceed Miles' 120s tick limit.
            # Cancellation leaves the engine offloaded before a health retry.
            "patch_cell_tick_timeout",
        )
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_LoRA_Recipe(MilesRecipe):
    """GLM-5.3-Flash rank-16 native LoRA, 32k context, on 16 H200s.

    Trains attention and dense/shared-expert MLP adapters; routed experts and
    absorbed DSA ``kv_b_proj`` stay frozen. The first run converts the frozen
    base on one 8-GPU node. Native adapter checkpoints can be resumed with
    ``lora_adapter_path`` on the same parallel layout; Bridge adapters are not
    compatible with this preset.
    """

    model_config_class: ClassVar[type[ModelConfig]] = GLM_5_3_Flash_LoRA
    miles_git_ref: str | None = "24bd5fb7166a185f309399e52f43b100e4369203"
    sglang_git_ref: str | None = "3a938213c1b349e1a8df0b5fdc57ef36a41c743b"
    image_run_commands: list[str] = field(default_factory=_image_commands)
    miles_model_name: str = "glm5.3-flash"
    model_name: str = "glm5_next"
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/GLM-5.3-Flash_native_torch_dist"

    use_session_server: bool | str = True
    tito_model: str = "glm53"
    pause_generation_mode: str = "in_place"

    # Convert the frozen FP8 base once on one 8-GPU node; training reshards it.
    conversion_tensor_model_parallel_size: int | None = 2
    conversion_pipeline_model_parallel_size: int | None = 1
    conversion_expert_model_parallel_size: int | None = 8
    conversion_expert_tensor_parallel_size: int | None = 1
    convert_ephemeral_disk_mb: int | None = 2 * 1024 * 1024

    gpu_type: str = "H200"
    memory: tuple[int, int] = (1024, 2 * 1024 * 1024)
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 8
    expert_model_parallel_size: int = 16
    sequence_parallel: bool = True

    lora_rank: int | None = 16
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    lora_adapter_path: str | None = None
    target_modules: str | None = "attn,mlp"
    # Native LoRA has no routed-expert adapters; absorbed DSA kv_b never runs.
    exclude_modules: str = "*.mlp.experts.*,*.self_attn.kv_b_proj"
    lora_base_cpu_backup: bool = True
    check_lora_weight_equal: bool = True
    no_gradient_accumulation_fusion: bool = True
    dsa_attention_backend: str = "tilelang"
    offload_train: bool = True
    offload_rollout: bool = True
    offload_train_target: str = "cpu"
    update_weight_buffer_size: int = 1024**3

    # Upstream native-LoRA DAPO batch, divisible by the two DP ranks.
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    # The inference window includes both the prompt and its response.
    rollout_max_prompt_len: int = 4096
    rollout_max_context_len: int = 32768
    rollout_max_response_len: int = 28672
    rm_type: str | None = "math"
    balance_data: bool = True
    skip_eval_before_train: bool = True
    use_rollout_routing_replay: bool = True
    lr: float = 1e-5
    calculate_per_token_loss: bool = True
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    seq_length: int = 32768
    max_tokens_per_gpu: MicroBatchSize = 32768
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    rollout_num_gpus_per_engine: int = 8
    sglang_tp_size: int = 8
    sglang_ep_size: int = 8
    sglang_dp_size: int | None = 1
    sglang_context_length: int = 32768
    sglang_mem_fraction_static: float = 0.5
    sglang_lora_backend: str | None = "triton"
    sglang_max_lora_rank: int = 16
    sglang_max_loras_per_batch: int = 1
    sglang_moe_runner_backend: str | None = "triton"
    sglang_disable_shared_experts_fusion: bool = True
    sglang_chunked_prefill_size: int = 8192
    sglang_disable_radix_cache: bool = True
    sglang_dsa_prefill_backend: str = "tilelang"
    sglang_dsa_decode_backend: str = "tilelang"
    sglang_kv_cache_dtype: str = "bfloat16"
    router_health_success_threshold: int = 1
    router_health_check_interval_secs: int = 15
    router_health_failure_threshold: int = 40
    rollout_health_check_interval: int = 300
    rollout_health_check_timeout: int = 300
    rollout_health_check_first_wait: int = 1800
    distributed_timeout_minutes: int = 60

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root:/root/miles:/root/Megatron-LM",
            "CONVERT_LOCAL_STAGING": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_MNNVL_ENABLE": "0",
            "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
            "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
            "PYTHONFAULTHANDLER": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TRITON_CACHE_DIR": "/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        }
    )

    @model_validator(mode="after")
    def _check_adapter_rank(self) -> "GLM_5_3_Flash_LoRA_Recipe":
        if self.lora_rank is None or self.lora_rank <= 0:
            raise ValueError("GLM LoRA requires a positive lora_rank")
        if self.sglang_max_lora_rank != self.lora_rank:
            raise ValueError("sglang_max_lora_rank must match lora_rank")
        return self

    @model_validator(mode="after")
    def _keep_image_commands(self) -> "GLM_5_3_Flash_LoRA_Recipe":
        required = _image_commands()
        current = self.image_run_commands or []
        if current[: len(required)] != required:
            object.__setattr__(
                self,
                "image_run_commands",
                [
                    *required,
                    *(c for c in current if c not in required),
                ],
            )
        return self
