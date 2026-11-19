"""Inkling 2 Small LoRA with response-heavy 256k coding trajectories."""

from dataclasses import field
from pathlib import Path

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.inkling_2_small import (
    Inkling_2_Small_LoRA_Recipe,
)

_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "frameworks/miles/modal_helpers/patches"
)


def _image_commands() -> list[str]:
    return [
        f"echo {encode_patch(name, _PATCH_DIR)} | base64 -d | python3"
        for name in ("patch_glm_5_3_flash_moe_seq_chunk", "patch_inkling2_long_context")
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Inkling_2_Small_LoRA_256k_Recipe(Inkling_2_Small_LoRA_Recipe):
    """Experimental 256k LoRA preset, one node of eight B300 GPUs.

    Reserves 16k initial prompt tokens and 240k response tokens. A coding
    rollout must supply exact token IDs, logprobs, tool-observation loss masks
    via ``custom_generate_function`` for multi-turn agents, and executable-test
    rewards via ``custom_rm_function``. There is no built-in math reward.
    Defaults to one proof step; full-length GPU validation is pending.
    """

    image_run_commands: list[str] = field(default_factory=_image_commands)
    custom_model_provider_path: str = (
        "modal_dojo.frameworks.miles.inkling2_long_context.model_provider"
    )
    max_seq_len: int = 262144
    rollout_max_prompt_len: int = 16384
    rollout_max_response_len: int = 245760
    eval_max_response_len: int = 245760
    seq_length: int = 262144
    sglang_context_length: int = 262144

    # Eight DP ranks, one GRPO group. Do not pack several long trajectories.
    rollout_batch_size: int = 1
    global_batch_size: int = 8
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    max_tokens_per_gpu: MicroBatchSize = 262144
    rm_type: str | None = None
    sglang_max_running_requests: int | None = 1
    sglang_server_concurrency: int | None = 8
    sglang_chunked_prefill_size: int = 4096
    capture_trace: bool = True

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
        if value("rm_type") == "math":
            raise ValueError(
                "Use executable-test rewards for the Inkling coding recipe"
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
