"""Beam LoRA with a 224k prompt and 32k response budget."""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from .beam import _BeamLoRABase, _image_commands


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Beam_1_0_501B_A23B_LoRA_256k_Recipe(_BeamLoRABase):
    """256k Beam LoRA on 16 training and 16 rollout H200s.

    Uses FlashAttention, CP2, and full activation recomputation.
    """

    patch_files: list[str] = field(
        default_factory=lambda: [
            str(
                Path(__file__).parents[2]
                / "frameworks/miles/modal_helpers/patches"
                / name
            )
            for name in (
                "patch_beam_cp.py",
                "patch_beam_lora_memory.py",
                "patch_beam_activation.py",
                "patch_beam_checkpoint.py",
            )
        ]
    )
    image_run_commands: list[str] = field(
        default_factory=lambda: [
            *_image_commands(),
            "python /tmp/patch_beam_cp.py",
            "python /tmp/patch_beam_lora_memory.py",
            "python /tmp/patch_beam_activation.py",
            "python /tmp/patch_beam_checkpoint.py",
        ]
    )

    _BEAM_ATTENTION_BACKENDS: ClassVar[tuple[str, ...]] = ("flash",)
    train_env_vars: dict | None = field(
        default_factory=lambda: {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    )
    # Save each completed update so an interruption loses at most one update.
    save_interval: int | None = 1
    max_retries: int = 10
    lora_adapter_path: str | None = None
    attention_backend: str = "flash"
    context_parallel_size: int = 2
    # Preserve TP8 sequence-parallel alignment after splitting tokens across CP2.
    data_pad_size_multiplier: int = 2
    cp_comm_type: str = "all_gather"
    rollout_max_context_len: int = 262144
    rollout_max_prompt_len: int = 229376
    rollout_max_response_len: int = 32768
    eval_max_response_len: int = 32768
    seq_length: int = 262144
    # The pinned worker subtracts one slot, then the scheduler reserves another.
    # Keep the actual prompt+response at 262144 while leaving both internal slots.
    sglang_context_length: int = 262146
    # Let SGLang admit requests according to KV capacity and retract when needed.
    sglang_max_running_requests: int | None = None
    sglang_cuda_graph_max_bs_decode: int = 64
    sglang_chunked_prefill_size: int = 4096

    @model_validator(mode="after")
    def _check_long_context(self) -> "Beam_1_0_501B_A23B_LoRA_256k_Recipe":
        overrides = self._escape_hatch_values()

        def effective(name):
            return overrides.get(name, getattr(self, name, None))

        for name in (
            "load",
            "lora_adapter_path",
            "start_rollout_id",
            "no_load_optim",
            "no_save_optim",
        ):
            if name in overrides:
                raise ValueError(
                    f"Set {name} on the Beam recipe directly so resume cannot be overridden by extra_config"
                )
        if overrides.get("fully_async"):
            raise ValueError(
                "Beam checkpoint resume supports the default one-batch async prefetch, not fully_async"
            )
        if self.max_retries and (not effective("save") or effective("no_save_optim")):
            raise ValueError(
                "Beam retries require checkpoint and optimizer saving; otherwise set max_retries=0"
            )
        cp_comm_type = effective("cp_comm_type")
        if isinstance(cp_comm_type, list) and len(cp_comm_type) == 1:
            cp_comm_type = cp_comm_type[0]
        if cp_comm_type != "all_gather":
            raise ValueError(
                "Beam sliding-window CP requires all_gather in this preset"
            )
        if effective("context_parallel_size") < 2:
            raise ValueError("The 256k Beam preset requires context_parallel_size >= 2")
        if (
            effective("sglang_context_length")
            < effective("rollout_max_context_len") + 2
        ):
            raise ValueError("Beam's pinned SGLang requires two extra context slots")
        return self
