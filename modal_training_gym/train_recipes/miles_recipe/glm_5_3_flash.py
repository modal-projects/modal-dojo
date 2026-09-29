"""GLM-5.3-Flash text GRPO, based on radixark/miles#2786."""

from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models.glm_5_3_flash import GLM_5_3_Flash
from modal_training_gym.common.models.base import ModelConfig
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe

_TRAIN_DISK_MIB = 2048 * 1024
_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)


def _image_patches() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in ("patch_glm_5_3_flash_kda", "patch_glm_5_3_flash_timing")
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_Recipe(MilesRecipe):
    """GLM-5.3-Flash text GRPO on four nodes of eight B300 GPUs.

    Port of upstream's 32-GPU layout; live validation on Modal is pending.
    Vision and MTP weights are excluded from training.
    """

    model_config_class: ClassVar[type[ModelConfig]] = GLM_5_3_Flash

    # glm53next supplies the matching SGLang and Megatron implementations.
    docker_image: str = (
        "radixark/miles:glm53next@sha256:"
        "66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162"
    )
    miles_git_ref: str | None = "cc76e23915b2132ecfff65b97331fd83b02637e6"
    image_run_commands: list[str] = field(default_factory=_image_patches)
    miles_model_name: str = "glm5.3-flash"
    model_name: str = "glm5_next"

    gpu_type: str = "B300"
    memory: tuple[int, int] = (1024, 2 * 1024 * 1024)
    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 8
    pipeline_model_parallel_size: int = 4
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = True
    decoder_first_pipeline_num_layers: int = 11
    decoder_last_pipeline_num_layers: int = 12

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "CONVERT_KEEP_PP1": "1",
            "NCCL_MNNVL_ENABLE": "0",
            "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
            # Inductor's subprocess pool deadlocks under torch_memory_saver.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )

    # The raw converter combines the three mHC alpha parameters on weight sync.
    megatron_to_hf_mode: str = "raw"
    ref_load: str = "/checkpoints/GLM-5.3-Flash_torch_dist"
    conversion_tensor_model_parallel_size: int = 8
    conversion_pipeline_model_parallel_size: int = 1
    conversion_expert_model_parallel_size: int = 8
    conversion_expert_tensor_parallel_size: int = 1
    convert_ephemeral_disk_mb: int | None = 1024 * 1024

    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    global_batch_size: int = 32
    rollout_temperature: float = 0.8
    rollout_max_response_len: int = 4096
    rm_type: str | None = "math"
    balance_data: bool = True
    skip_eval_before_train: bool = True
    use_dynamic_batch_size: bool = False
    micro_batch_size: int = 1
    max_tokens_per_gpu: int = 8192
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    bf16: bool = True
    use_distributed_optimizer: bool = True

    offload_train: bool = True
    offload_train_target: str = "disk"
    offload_train_disk_dir: str = "/tmp/train_offload"
    train_memory_margin_bytes: int = 3 * 1024**3
    update_weight_buffer_size: int = 1024**3
    no_save_optim: bool = True
    no_load_optim: bool = True
    no_save_rng: bool = True
    no_load_rng: bool = True
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": _TRAIN_DISK_MIB}
    )

    rollout_num_gpus_per_engine: int = 8
    sglang_device: str = "cuda"
    sglang_tp_size: int = 8
    sglang_ep_size: int = 8
    sglang_moe_runner_backend: str = "triton"
    sglang_mem_fraction_static: float = 0.7
    sglang_chunked_prefill_size: int = 8192
    sglang_disable_radix_cache: bool = True
    sglang_dsa_prefill_backend: str = "tilelang"
    sglang_dsa_decode_backend: str = "tilelang"
    sglang_kv_cache_dtype: str = "bfloat16"
    check_weight_update_equal: bool = True
    check_weight_update_skip_list: list[str] = field(
        default_factory=lambda: ["visual."]
    )
    router_health_success_threshold: int = 1
    router_health_check_interval_secs: int = 15
    router_health_failure_threshold: int = 40
    rollout_health_check_interval: int = 300
    rollout_health_check_timeout: int = 300
    rollout_health_check_first_wait: int = 1800
    distributed_timeout_minutes: int = 60

    @model_validator(mode="after")
    def _keep_disk_reservation(self) -> "GLM_5_3_Flash_Recipe":
        kwargs = self.train_function_kwargs or {}
        if "ephemeral_disk" not in kwargs:
            object.__setattr__(
                self,
                "train_function_kwargs",
                {"ephemeral_disk": _TRAIN_DISK_MIB, **kwargs},
            )
        return self

    @model_validator(mode="after")
    def _keep_image_patches(self) -> "GLM_5_3_Flash_Recipe":
        patches = _image_patches()
        current = self.image_run_commands or []
        if current[: len(patches)] != patches:
            object.__setattr__(
                self,
                "image_run_commands",
                [*patches, *(c for c in current if c not in patches)],
            )
        return self
