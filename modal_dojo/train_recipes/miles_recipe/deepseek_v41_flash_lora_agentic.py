"""DeepSeek V4.1 Flash native LoRA for multi-turn agent rollouts."""

from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash_lora import (
    DeepSeek_V4_1_Flash_LoRA_Recipe,
    _PATCH_DIR,
)


def _agentic_extra_config() -> dict:
    return {
        **DeepSeek_V4_1_Flash_LoRA_Recipe().extra_config,
        # The rollout adapter supplies a stable session ID. Assign new
        # conversations by load and retain their engine's prefix cache.
        "sglang_router_policy": "manual",
        "router_assignment_mode": "min_load",
    }


def _agentic_environment() -> dict[str, str]:
    return {
        **DeepSeek_V4_1_Flash_LoRA_Recipe().environment,
        "DOJO_UPDATE_DIAGNOSTICS": "1",
        "DOJO_UPDATE_PROBE_TOKENS": "1536",
    }


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe(DeepSeek_V4_1_Flash_LoRA_Recipe):
    """Multi-turn 32K LoRA: 30K history, 2K per call, 32 responses on 32 H200s.

    Supply the workload's rollout/reward callbacks and dataset separately.
    Session IDs must remain stable across turns for manual routing. The runtime
    supports fixed one-fragment microbatches and loss-masked scheduling padding.

    Saves optimizer/RNG state for continuation and enables sampled before/after
    policy diagnostics. Frozen-base disk snapshot reuse is an opt-in prototype:
    allocator correctness has been tested, full-model speedup has not.
    """

    _SKIP_FIELDS: ClassVar[frozenset[str]] = (
        DeepSeek_V4_1_Flash_LoRA_Recipe._SKIP_FIELDS | {"reuse_frozen_base_backup"}
    )
    name: str = "deepseek-v41-flash-lora-agentic"
    rollout_max_prompt_len: int = 30720
    rollout_max_response_len: int = 2048
    num_rollout: int = 2
    save_interval: int | None = 1
    skip_eval_before_train: bool = False
    dynamic_sampling_filter_path: str = ""
    no_save_optim: bool = False
    no_load_optim: bool = False
    no_load_rng: bool = False
    max_retries: int = 0
    capture_trace: bool = True
    extra_config: dict | None = field(default_factory=_agentic_extra_config)
    environment: dict[str, str] = field(default_factory=_agentic_environment)
    reuse_frozen_base_backup: bool = False

    @model_validator(mode="after")
    def _configure_frozen_backup(self):
        flag = "DOJO_REUSE_FROZEN_BASE_BACKUP"
        if not self.reuse_frozen_base_backup:
            if self.environment.get(flag) == "1":
                raise ValueError(
                    "Enable snapshot reuse with reuse_frozen_base_backup=True"
                )
            return self
        if (
            not self.lora_rank
            or (self.extra_config or {}).get("sglang_device") != "cuda"
        ):
            raise ValueError("Frozen-base snapshot reuse requires native CUDA LoRA")
        if not self.environment.get("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR"):
            raise ValueError(
                "Frozen-base snapshot reuse requires disk-backed serving weights"
            )
        command = (
            f"echo {encode_patch('patch_sglang_frozen_weight_backup', _PATCH_DIR)}"
            " | base64 -d | python3"
        )
        object.__setattr__(self, "environment", {**self.environment, flag: "1"})
        if command not in self.image_run_commands:
            object.__setattr__(
                self, "image_run_commands", [*self.image_run_commands, command]
            )
        return self
