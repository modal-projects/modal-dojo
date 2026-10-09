"""Spindle-backed training recipe.

``SpindleRecipe`` keeps the :class:`MilesRecipe` interface so a Miles config can
be swapped for a Spindle one without rewriting it, but trains through a
`Spindle <https://github.com/modal-projects/spindle>`_ deployment (a
Tinker-compatible API served from Modal) using the tinker-cookbook GRPO
algorithm (``tinker_cookbook.rl.train``) as the training loop.

Only the Qwen3.8-27B LoRA preset at a 32K context length is supported; see
:class:`modal_dojo.train_recipes.spindle_recipe.qwen3_8_27b.Qwen3_8_27B_Spindle_Recipe`.
"""

from __future__ import annotations

import warnings
from dataclasses import field
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

if TYPE_CHECKING:
    from modal_dojo.common.models import ModelConfig

SPINDLE_GIT_URL = "https://github.com/modal-projects/spindle.git"
# https://github.com/modal-projects/spindle/pull/35
SPINDLE_GIT_REF = "pull/35/head"
TINKER_COOKBOOK_GIT_URL = "https://github.com/thinking-machines-lab/tinker-cookbook.git"
TINKER_COOKBOOK_GIT_REF = "main"

SUPPORTED_MODEL_NAME = "Qwen/Qwen3.8-27B"
SUPPORTED_CONTEXT_LENGTH = 32768
SUPPORTED_RM_TYPES = frozenset({"deepscaler", "math", "boxed"})

# Spindle-only fields; never emitted as Miles CLI flags.
_SPINDLE_SKIP = frozenset(
    {
        "max_context_length",
        "spindle_git_ref",
        "spindle_frontend",
        "spindle_environment",
        "spindle_api_secret",
        "spindle_proxy_secret",
        "spindle_platform_overrides",
        "spindle_config_overrides",
        "tinker_cookbook_git_ref",
        "grpo_config_overrides",
        "remove_constant_reward_groups",
    }
)

# Miles fields the Spindle launcher cannot honour. Setting them to a non-default
# value raises instead of being silently ignored.
_UNSUPPORTED_FIELDS = (
    "use_critic",
    "custom_generate_function",
    "custom_reward_post_process_function",
    "rollout_function",
    "custom_rollout_log_function",
    "custom_eval_rollout_log_function",
    "custom_megatron_before_log_prob_hook",
    "custom_megatron_before_train_step_hook",
    "num_epoch",
    "over_sampling_batch_size",
    "dynamic_sampling_filter_path",
    "local_miles",
    "miles_git_ref",
    "sglang_git_ref",
    "patch_files",
    "miles_model_script",
    "eval_config",
    "sglang_config",
    "train_env_vars",
    "multimodal_keys",
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class SpindleRecipe(MilesRecipe):
    """Miles-compatible recipe trained through Spindle with tinker-cookbook GRPO.

    Every :class:`MilesRecipe` field is accepted. The launcher maps them onto a
    Spindle deployment config (``spindle.configuration.BaseConfig``) and a
    tinker-cookbook ``rl.train.Config``:

    ========================================  ===========================================
    Miles field                               Spindle / GRPO setting
    ========================================  ===========================================
    ``gpu_type``                              ``trainer_gpu`` and ``inference_gpu``
    ``actor_num_nodes`` / ``*_gpus_per_node`` ``trainer_nodes`` / ``trainer_gpus_per_node``
    ``rollout_num_gpus`` / ``*_per_engine``   inference replicas / ``inference_gpus_per_node``
    ``tensor/pipeline/context_parallel_size`` ``miles_cfg`` parallelism
    ``max_tokens_per_gpu``                    ``miles_cfg.max_tokens_per_gpu``
    ``lora_rank`` / ``lora_alpha``            ``miles_cfg.max_lora_rank`` / ``default_lora_alpha``
    ``target_modules``                        ``miles_cfg.target_modules``
    ``recompute_*``                           ``miles_cfg.cli_options``
    ``sglang_*``                              ``sglang_cfg``
    ``environment``                           ``trainer_env``
    ``num_rollout``                           GRPO ``max_steps``
    ``rollout_batch_size``                    prompt groups per step
    ``n_samples_per_prompt``                  GRPO group size
    ``global_batch_size``                     ``num_substeps`` (samples per step / batch)
    ``rollout_max_response_len``              ``max_tokens``
    ``rollout_temperature``                   ``temperature``
    ``lr``                                    ``learning_rate``
    ``kl_coef`` / ``kl_loss_coef``            ``kl_penalty_coef``
    ``eps_clip`` / ``eps_clip_high``          ``loss_fn="ppo"`` clip bounds when set
    ``save_interval`` / ``eval_interval``     ``save_every`` / ``eval_every``
    ``custom_rm_function``                    environment reward (Miles ``rm(args, sample)``)
    ``apply_chat_template_kwargs``            tokenizer ``apply_chat_template`` kwargs
    ========================================  ===========================================

    Args:
        max_context_length:
            Spindle deployment context length. Only 32768 is supported.
        spindle_git_ref:
            Spindle git ref installed into the deployment image.
        spindle_frontend:
            Name of the Spindle frontend app. Defaults to ``spindle-<recipe name>``.
        spindle_environment:
            Modal environment the Spindle apps deploy into. Defaults to the
            current environment.
        spindle_api_secret:
            Modal Secret holding ``TINKER_API_KEY`` for the Spindle frontend.
        spindle_proxy_secret:
            Modal Secret holding the sampler proxy credentials.
        spindle_platform_overrides:
            Dotted overrides for ``BaseConfig.platform`` (e.g. ``{"storage.checkpoints": ...}``).
        spindle_config_overrides:
            Dotted overrides applied last to the generated Spindle config
            (e.g. ``{"sglang_cfg.schedule_policy": "fcfs"}``).
        tinker_cookbook_git_ref:
            tinker-cookbook git ref installed into the GRPO driver image.
        grpo_config_overrides:
            Extra ``tinker_cookbook.rl.train.Config`` fields applied last.
        remove_constant_reward_groups:
            Drop groups whose samples all share one reward (GRPO has no signal there).
    """

    # ── Spindle deployment ────────────────────────────────────────────────────
    max_context_length: int = SUPPORTED_CONTEXT_LENGTH
    spindle_git_ref: str = SPINDLE_GIT_REF
    spindle_frontend: str = ""
    spindle_environment: str | None = None
    spindle_api_secret: str = "spindle-api"
    spindle_proxy_secret: str = "spindle-proxy"
    spindle_platform_overrides: dict[str, Any] = field(default_factory=dict)
    spindle_config_overrides: dict[str, Any] = field(default_factory=dict)

    # ── GRPO driver ───────────────────────────────────────────────────────────
    tinker_cookbook_git_ref: str = TINKER_COOKBOOK_GIT_REF
    grpo_config_overrides: dict[str, Any] = field(default_factory=dict)
    remove_constant_reward_groups: bool = False

    # ── Miles defaults re-based for a LoRA Spindle trainer ────────────────────
    gpu_type: str = "H200"
    colocate: bool = False
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = 8
    rollout_num_gpus_per_engine: int = 1
    tensor_model_parallel_size: int = 4
    lora_rank: int | None = 32
    lora_alpha: int | None = 32
    target_modules: str | None = (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    max_tokens_per_gpu: int = SUPPORTED_CONTEXT_LENGTH
    sglang_mem_fraction_static: float = 0.8
    sglang_max_running_requests: int | None = 32
    lr: float = 4e-5
    max_retries: int = 0

    _SKIP_FIELDS: ClassVar[frozenset[str]] = MilesRecipe._SKIP_FIELDS | _SPINDLE_SKIP
    sft_supported: ClassVar[bool] = False

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_spindle_support(self) -> "SpindleRecipe":
        if self.max_context_length != SUPPORTED_CONTEXT_LENGTH:
            raise DojoConfigError(
                "SpindleRecipe supports only max_context_length="
                f"{SUPPORTED_CONTEXT_LENGTH} (got {self.max_context_length})"
            )
        if self.loss_type != "policy_loss":
            raise DojoConfigError("SpindleRecipe supports only loss_type='policy_loss'")
        if self.rm_type is not None and self.rm_type not in SUPPORTED_RM_TYPES:
            raise DojoConfigError(
                f"SpindleRecipe supports rm_type in {sorted(SUPPORTED_RM_TYPES)} "
                f"(the built-in boxed-answer reward) or custom_rm_function, got {self.rm_type!r}"
            )
        if not self.lora_rank:
            raise DojoConfigError(
                "SpindleRecipe trains LoRA adapters; set lora_rank to a positive integer"
            )
        if self.rollout_max_response_len >= self.max_context_length:
            raise DojoConfigError(
                f"rollout_max_response_len={self.rollout_max_response_len} must leave "
                f"room for the prompt within max_context_length={self.max_context_length}"
            )
        samples = self.rollout_batch_size * self.n_samples_per_prompt
        if samples % self.global_batch_size != 0:
            raise DojoConfigError(
                f"rollout_batch_size * n_samples_per_prompt ({samples}) must be a "
                f"multiple of global_batch_size ({self.global_batch_size})"
            )
        defaults = MilesRecipe()
        unsupported = sorted(
            name
            for name in _UNSUPPORTED_FIELDS
            if getattr(self, name) != getattr(defaults, name)
        )
        if unsupported:
            raise DojoConfigError(
                "SpindleRecipe does not support: " + ", ".join(unsupported)
            )
        if self.advantage_estimator != "grpo":
            warnings.warn(
                f"SpindleRecipe always trains with GRPO; advantage_estimator="
                f"{self.advantage_estimator!r} is ignored.",
                stacklevel=2,
            )
        return self

    # ── Model presets ─────────────────────────────────────────────────────────

    @classmethod
    def get_base_recipe(cls, model_config: "ModelConfig") -> "SpindleRecipe":
        from modal_dojo.train_recipes.spindle_recipe.qwen3_8_27b import (
            Qwen3_8_27B_Spindle_Recipe,
        )

        if model_config.model_name == SUPPORTED_MODEL_NAME:
            return Qwen3_8_27B_Spindle_Recipe()
        raise DojoConfigError(
            f"SpindleRecipe supports only {SUPPORTED_MODEL_NAME} at "
            f"{SUPPORTED_CONTEXT_LENGTH} context; got {model_config.model_name!r}"
        )

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        if model.model_name != SUPPORTED_MODEL_NAME:
            raise DojoConfigError(
                f"SpindleRecipe supports only {SUPPORTED_MODEL_NAME}; "
                f"got {model.model_name!r}"
            )
        super().validate_model_parallelism(model)

    # ── Derived GRPO settings ─────────────────────────────────────────────────

    @property
    def grpo_num_substeps(self) -> int:
        """Optimizer steps per GRPO iteration implied by ``global_batch_size``."""
        return (self.rollout_batch_size * self.n_samples_per_prompt) // (
            self.global_batch_size
        )

    @property
    def grpo_kl_penalty_coef(self) -> float:
        if self.kl_coef:
            return self.kl_coef
        if self.use_kl_loss and self.kl_loss_coef:
            return self.kl_loss_coef
        return 0.0

    @property
    def inference_replicas(self) -> int:
        """SGLang replicas implied by the rollout GPU budget."""
        gpus = self.rollout_num_gpus or self.rollout_num_gpus_per_engine
        return max(1, gpus // self.rollout_num_gpus_per_engine)

    def spindle_deployment_name(self, training_run_id: str = "") -> str:
        return self.name or (
            f"dojo-{training_run_id}" if training_run_id else "dojo-spindle"
        )

    def spindle_frontend_name(self, training_run_id: str = "") -> str:
        return self.spindle_frontend or (
            f"spindle-{self.spindle_deployment_name(training_run_id)}"
        )
