"""Inkling 2 Small LoRA with response-heavy 256k coding trajectories.

The BF16 trainer and the engine serving the released NVFP4/MXFP8 experts do not
fit on the GPUs together, so the trainer is parked on local disk during rollouts
and only the adapter is synced.
"""

from dataclasses import field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.models import Inkling_2_Small, ModelConfig
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)


def _image_commands() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in ("patch_glm_5_3_flash_moe_seq_chunk", "patch_inkling2_long_context")
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Small_LoRA_256k_Recipe(MilesRecipe):
    """Rank-32 LoRA GRPO, 262,144-token context, 1 node × 8 B300 GPUs.

    Reserves 16k initial prompt tokens and 240k response tokens. Multi-turn
    coding rollouts must supply exact token IDs, logprobs and tool-observation
    loss masks via ``custom_generate_function``, plus executable-test rewards
    via ``custom_rm_function``. Defaults to one proof step and no reward scorer;
    full-length GPU validation is pending.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Inkling_2_Small
    # Copy of radixark/miles:inkling-2-small, same digest: Miles radixark/miles#4819
    # with an SGLang build that serves the model. Its Miles and SGLang trees are
    # not git checkouts, so the git-ref overlays cannot be used on it.
    # TODO: Use a released image once Miles #4819 and its SGLang support land.
    docker_image: str = (
        "modalresearch/miles:dojo-inkling-2-small@sha256:"
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

    # Eight DP ranks, one GRPO group. Do not pack several long trajectories.
    rollout_batch_size: int = 1
    n_samples_per_prompt: int = 8
    global_batch_size: int = 8
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    balance_data: bool = True
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
    max_tokens_per_gpu: MicroBatchSize = 262144
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
    sglang_context_length: int = 262144
    sglang_max_running_requests: int | None = 1
    sglang_server_concurrency: int | None = 8
    sglang_chunked_prefill_size: int = 4096
    max_seq_len: int = 262144
    rollout_max_prompt_len: int = 16384
    rollout_max_response_len: int = 245760
    eval_max_response_len: int = 245760
    seq_length: int = 262144
    capture_trace: bool = True
    # The pinned Miles image parses Inkling reasoning/tool blocks itself.
    tito_model: str = "inkling"
    use_session_server: str = "v1"
    image_run_commands: list[str] = field(default_factory=_image_commands)
    custom_model_provider_path: str = (
        "modal_dojo.frameworks.miles.inkling2_long_context.model_provider"
    )
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

    @model_validator(mode="after")
    def _validate_streaming_backend(self) -> "Inkling_2_Small_LoRA_256k_Recipe":
        overrides = self._escape_hatch_values()

        def value(name, default=None):
            return overrides.get(name, getattr(self, name, default))

        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ):
            if value(name) != 1:
                raise ValueError(f"Upstream Inkling 2 requires {name}=1")
        if value("sequence_parallel"):
            raise ValueError("Upstream Inkling 2 does not support sequence_parallel")
        if value("loss_type") != "policy_loss" or value("qkv_format") != "thd":
            raise ValueError("Inkling 256k requires packed thd policy_loss")
        for name in (
            "recompute_loss_function",
            "true_on_policy_mode",
            "use_sampling_support_replay",
        ):
            if value(name, False):
                raise ValueError(f"Inkling streaming scores do not support {name}")
        if (
            value("recompute_granularity") != "full"
            or value("recompute_method") != "uniform"
            or value("recompute_num_layers") != 1
        ):
            raise ValueError(
                "Inkling 256k requires full recomputation of one layer at a time"
            )
        if value("use_dynamic_batch_size") or value("micro_batch_size") != 1:
            raise ValueError(
                "Inkling 256k requires micro_batch_size=1 without dynamic packing"
            )
        required = _image_commands()
        if (self.image_run_commands or [])[: len(required)] != required:
            object.__setattr__(
                self,
                "image_run_commands",
                [
                    *required,
                    *(
                        command
                        for command in self.image_run_commands or []
                        if command not in required
                    ),
                ],
            )
        return self
