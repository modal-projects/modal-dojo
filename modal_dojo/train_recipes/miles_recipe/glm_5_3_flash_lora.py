"""Experimental bridge-mode GLM-5.3-Flash LoRA from radixark/miles#3098."""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.glm_5_3_flash import GLM_5_3_Flash_LoRA
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)
_BRIDGE_REF = "6527b18e8bb0db994a267e6dfd4db7dafc669df9"
_TARGETS = (
    "self_attention.linear_q",
    "self_attention.linear_k",
    "self_attention.linear_v",
    "self_attention.linear_b",
    "self_attention.linear_f_a",
    "self_attention.linear_f_b",
    "self_attention.linear_g_a",
    "self_attention.linear_g_b",
    "self_attention.linear_q_down_proj",
    "self_attention.linear_q_up_proj",
    "self_attention.linear_kv_down_proj",
    "self_attention.linear_kv_up_proj",
    "self_attention.linear_proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
    "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.linear_fc2",
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
)


def _image_commands() -> list[str]:
    return [
        # Keep the image's CUDA/PyTorch/Megatron stack; replace only Bridge.
        "python3 -m pip install --no-deps --no-build-isolation "
        f"'megatron-bridge @ git+https://github.com/radixark/Megatron-Bridge.git@{_BRIDGE_REF}'",
        "python3 -m pip install --no-deps sglang-kernel==0.4.6.post1",
        # Replace best-effort shared timing instrumentation with the adapter
        # for this pinned driver. Other reporting files remain patched.
        "cd /root/miles && git checkout HEAD -- train.py "
        "miles/backends/megatron_utils/actor.py miles/backends/megatron_utils/model.py",
        *[
            f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
            for name in (
                "patch_glm_5_3_flash_kda",
                "patch_glm_5_3_flash_lora_kda_backward",
                "patch_glm_5_3_flash_lora_timing",
                "patch_glm_5_3_flash_tito",
            )
        ],
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_5_3_Flash_LoRA_Recipe(MilesRecipe):
    """GLM-5.3-Flash rank-16 bridge LoRA on three nodes of eight H200 GPUs."""

    model_config_class: ClassVar[type[ModelConfig]] = GLM_5_3_Flash_LoRA
    docker_image: str = (
        "radixark/miles:glm53next@sha256:"
        "66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162"
    )
    miles_git_ref: str | None = "5a5353d36c9133958f9ddb62c93be463bc849f88"
    sglang_git_ref: str | None = "94c97e3a7d7ab1dcbba82f4c09046d2c209b1aa7"
    image_run_commands: list[str] = field(default_factory=_image_commands)
    miles_model_name: str = "glm5.3-flash"
    model_name: str = "glm5_next"

    use_session_server: bool | str = True
    tito_model: str = "glm53"

    gpu_type: str = "H200"
    memory: tuple[int, int] = (1024, 2 * 1024 * 1024)
    actor_num_nodes: int = 3
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 8
    expert_model_parallel_size: int = 24
    sequence_parallel: bool = True

    lora_rank: int | None = 16
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    target_modules: str | None = ",".join(f"decoder.layers.*.{t}" for t in _TARGETS)
    lora_base_cpu_backup: bool = True
    check_lora_weight_equal: bool = True
    no_gradient_accumulation_fusion: bool = True
    dsa_attention_backend: str = "tilelang"
    offload_train: bool = True
    offload_rollout: bool = True
    offload_train_target: str = "cpu"
    update_weight_buffer_size: int = 1024**3

    # 15 prompts x 4 samples = 60, which divides the three DP ranks.
    rollout_batch_size: int = 15
    n_samples_per_prompt: int = 4
    global_batch_size: int = 60
    rollout_max_context_len: int = 32768
    rollout_max_response_len: int = 32768
    rm_type: str | None = "math"
    balance_data: bool = True
    skip_eval_before_train: bool = True
    use_rollout_routing_replay: bool = True
    lr: float = 1e-5
    calculate_per_token_loss: bool = True
    micro_batch_size: int = 1
    seq_length: int = 32768
    max_tokens_per_gpu: int = 32768
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
            "PYTHONPATH": "/root/Megatron-LM/",
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
