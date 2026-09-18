import os
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import Any, Literal

from modal_training_gym.train_recipes.base import (
    BaseTrainRecipe,
    RecipeType,
    # Re-exported for backwards compatibility (e.g. frameworks/slime/launcher.py
    # imports the volume paths from this module).
    CHECKPOINTS_PATH as CHECKPOINTS_PATH,
    DATA_PATH as DATA_PATH,
    HF_CACHE_PATH as HF_CACHE_PATH,
    JSON_CONFIG_FIELDS as JSON_CONFIG_FIELDS,
)
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import (
    ModelArchitecture,
    ModelConfig,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.train_recipes.gpu_allocation import (
    validate_num_experts_divisible_by_expert_parallel_size,
    resolve_gpu_allocation,
    validate_megatron_actor_parallelism,
)

import modal

# ── Types ─────────────────────────────────────────────────────────────────────

_SLIME_SKIP = {
    "recipe_type",
    "environment",
    "async_mode",
    "physical_gpus_per_node",
    "wandb",
    "name",
    "app_tags",
    "capture_trace",
    "trace_sample_limit",
    "image_overlay",
    "image_overlay_source_roots",
    "local_slime",
    "memory",
    "cloud",
    "region",
    "checkpoint",
    "custom_rm_function",
    "custom_generate_function",
    "custom_rollout_log_function",
    "custom_eval_rollout_log_function",
    "rollout_function",
    "custom_megatron_before_log_prob_hook",
    "custom_megatron_before_train_step_hook",
    "sglang_request_params",
    "slime_model_script",
    "source_hf_checkpoint",
    "megatron_conversion_hf_checkpoint",
    "patch_files",
    "image_run_commands",
    "image_env",
    "train_function_kwargs",
    "attempt_mode",
    "max_retries",
    "max_attempts",
    "conversion_pipeline_model_parallel_size",
    "conversion_tensor_model_parallel_size",
    "conversion_expert_model_parallel_size",
    "conversion_expert_tensor_parallel_size",
}

YAML_CONFIG_FIELDS = ("eval_config", "extra_config", "sglang_config")

_HOOK_PATH_CONFIG_KEYS = {
    "custom_rollout_log_function": "training_gym_custom_rollout_log_function_path",
    "custom_eval_rollout_log_function": "training_gym_custom_eval_rollout_log_function_path",
    "custom_megatron_before_log_prob_hook": "training_gym_custom_megatron_before_log_prob_hook_path",
    "custom_megatron_before_train_step_hook": "training_gym_custom_megatron_before_train_step_hook_path",
}

_HOOK_WRAPPER_PATHS = {
    "custom_rollout_log_function": "modal_training_gym.frameworks.slime.phase_reporting.log_rollout_data",
    "custom_eval_rollout_log_function": "modal_training_gym.frameworks.slime.phase_reporting.log_eval_rollout_data",
    "custom_megatron_before_log_prob_hook": "modal_training_gym.frameworks.slime.phase_reporting.before_log_prob_hook",
    "custom_megatron_before_train_step_hook": "modal_training_gym.frameworks.slime.phase_reporting.before_train_step_hook",
}


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class SlimeRecipe(BaseTrainRecipe):
    """Recipe dataclass for configuring slime GRPO training on Modal.

    Don't see the configuration flags that you need? You can pass them in the `extra_config` field.
    """

    # ── Required: cluster and parallelism ──────────────────────────────────
    gpu_type: str
    colocate: bool
    tensor_model_parallel_size: int
    sequence_parallel: bool
    rollout_num_gpus_per_engine: int

    # ── Required: rollout ──────────────────────────────────────────────────
    num_rollout: int
    rollout_batch_size: int  # This is rollout_tp_size
    rollout_max_response_len: int
    rollout_temperature: float

    # ── Required: checkpointing ────────────────────────────────────────────
    save_interval: int

    # ── App identity ─────────────────────────────────────────────────────────
    recipe_type: RecipeType = RecipeType.SLIME
    name: str = ""
    app_tags: dict = field(default_factory=dict)

    # ── Launcher instructions (not slime CLI flags) ─────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    wandb: WandbConfig | None = None
    image_overlay: Callable[[modal.Image], modal.Image] | None = None
    # Local files/directories copied by ``image_overlay`` whose content must be
    # bound into a committed attempt's scientific run contract. Paths are
    # never persisted; only stable labels and content hashes are recorded.
    image_overlay_source_roots: list[str] = field(default_factory=list)
    local_slime: str | None = None
    memory: int | tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    slime_model_script: str = ""
    source_hf_checkpoint: str | None = None
    megatron_conversion_hf_checkpoint: str | None = None
    patch_files: list[str] = field(default_factory=list)
    image_run_commands: list[str] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)
    train_function_kwargs: dict[str, Any] = field(default_factory=dict)
    # ``committed`` gives every Modal invocation an immutable save namespace and
    # permits resume only from a framework-authenticated boundary manifest.
    attempt_mode: Literal["legacy", "committed"] = "legacy"
    max_retries: int = 3
    # Independent of Modal's retry policy, cap the number of logical attempt
    # namespaces that may enter Ray/model initialization.  This is a second,
    # application-level guard for scientific jobs that must never continue in
    # a replacement attempt after a platform interruption.
    max_attempts: int | None = None

    # ── Per-sample execution tracing (dashboard timeline) ───────────────────
    # When True, the rollout recorder attaches slime's per-sample trace (the
    # generate/reward/tool-call timeline) to the first `trace_sample_limit`
    # samples of each rollout. Off by default — traces inflate payloads, so
    # sampling keeps the added volume well under 1%. Not a slime CLI flag.
    capture_trace: bool = False
    trace_sample_limit: int = 16

    # ── Cluster and parallelism (optional) ─────────────────────────────────
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    # Modal worker size; trainer ranks and Slime CLI flags remain independent.
    # None preserves the existing allocation for all other recipes.
    physical_gpus_per_node: int | None = None
    rollout_num_gpus: int | None = None
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None

    # ── RL algorithm ────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 2
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = False
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    calculate_per_token_loss: bool = False
    ref_load: str = ""

    # ── Dynamic sampling (DAPO) ────────────────────────────────────────────
    over_sampling_batch_size: int | None = None
    dynamic_sampling_filter_path: str | None = None
    balance_data: bool = False

    # ── Rollout (optional) ─────────────────────────────────────────────────
    rollout_shuffle: bool = True
    rollout_top_p: float = 1.0
    rollout_stop_token_ids: list[int] | None = None
    sglang_mem_fraction_static: float = 0.75

    # ── Training ────────────────────────────────────────────────────────────
    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    min_lr: float = 0.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"

    # ── Memory and precision ────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    use_distributed_optimizer: bool = False
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    # ── Dynamic batching ────────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216

    # QKV layout for the Megatron backend. Emitted as --qkv-format (slime's own
    # default is "thd"). Set explicitly because SLIME_IMAGE nightly-dev-20260701a's
    # compute_advantages_and_returns reads args.qkv_format and AttributeError's at
    # the first train step if it isn't provided.
    qkv_format: str = "thd"

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 16384
    eval_top_p: float = 1.0
    eval_config: dict | None = None

    # ── Checkpointing (optional) ───────────────────────────────────────────
    save: str = "/checkpoints"
    load: str = ""
    no_save_optim: bool = False
    megatron_to_hf_mode: str = ""
    use_fault_tolerance: bool = True

    # Regex patterns of parameter names to freeze (slime's
    # --freeze-params-name-list, matched with re.search). Used e.g. to freeze a
    # VL model's vision tower so RL only updates the language backbone.
    freeze_params_name_list: list[str] | None = None

    # ── Weight sync (megatron trainer → sglang rollout engines) ──────────
    # Default matches slime's own default. ``delta`` mode pin-snapshots the
    # last broadcast on CPU and ships only byte-level changes, which is
    # ~5-10× faster than ``full`` for large models where weights barely
    # move per rollout (e.g. 35B-class MoE). Pair with
    # ``update_weight_transport="disk"`` if the trainer and rollout engines
    # share a filesystem.
    update_weight_mode: str = "full"
    update_weight_transport: str = "nccl"
    update_weight_encoding: str = "indices"
    update_weight_disk_dir: str = ""

    # ── Reward model ─────────────────────────────────────────────────────────
    rm_type: str | None = None

    # -- Slime customization flags ───────────────────────────────────────────
    # See https://github.com/THUDM/slime/blob/0988f0f4a0ab55d1bb3ce6285a597d912144fa80/docs/en/get_started/customization.md#1-rollout-function---rollout-function-path
    custom_rm_function: Callable | None = None
    custom_generate_function: Callable | None = None
    custom_rollout_log_function: Callable | str | None = None
    custom_eval_rollout_log_function: Callable | str | None = None
    rollout_function: Callable | str | None = None
    custom_megatron_before_log_prob_hook: Callable | str | None = None
    custom_megatron_before_train_step_hook: Callable | str | None = None

    # ── SGLang rollout engine ──────────────────────────────────────────────
    sglang_enable_dp_attention: bool = False
    sglang_dp_size: int | None = None
    sglang_ep_size: int | None = None
    sglang_enable_dp_lm_head: bool = False
    sglang_disable_custom_all_reduce: bool = False
    sglang_cuda_graph_bs: list[int] | None = None
    sglang_max_running_requests: int | None = None
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None

    # ── SGLang / config overrides ───────────────────────────────────────────
    extra_config: dict | None = None
    sglang_config: dict | None = None
    sglang_request_params: dict | None = None
    apply_chat_template_kwargs: dict | str = ""
    train_env_vars: dict | str | None = None
    multimodal_keys: dict | str | None = None

    # ── Validators ───────────────────────────────────────────────────────────

    @staticmethod
    def _callable_path(fn: Callable) -> str:
        mod = getattr(fn, "__module__", None) or ""
        name = getattr(fn, "__qualname__", None) or fn.__name__
        if mod == "__main__":
            import inspect

            try:
                src_file = inspect.getfile(fn)
                if os.path.isfile(src_file):
                    mod = Path(src_file).stem
                else:
                    mod = "__pending__"
            except (TypeError, OSError):
                mod = "__pending__"
        return f"{mod}.{name}"

    @staticmethod
    def _path_or_callable_path(value: Callable | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return SlimeRecipe._callable_path(value)

    @model_validator(mode="after")
    def _resolve_callable_paths(self) -> "SlimeRecipe":
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be positive when provided")
        cfg = dict(self.extra_config) if isinstance(self.extra_config, dict) else {}
        if self.custom_generate_function is not None:
            if not cfg.get("custom_generate_function_path"):
                cfg["custom_generate_function_path"] = self._callable_path(
                    self.custom_generate_function
                )
        for field_name, config_key in _HOOK_PATH_CONFIG_KEYS.items():
            value = getattr(self, field_name)
            if value is None or cfg.get(config_key):
                continue
            if isinstance(value, str):
                cfg[config_key] = value
            else:
                cfg[config_key] = self._callable_path(value)
        if cfg != (self.extra_config or {}):
            object.__setattr__(self, "extra_config", cfg)
        return self

    @model_validator(mode="after")
    def _validate_gpu_allocation(self) -> "SlimeRecipe":
        resolve_gpu_allocation(self)
        validate_megatron_actor_parallelism(self)
        return self

    # ── Container → slime flag converters ────────────────────────────────────

    @classmethod
    def _dataset_to_fields(cls, ds: "DatasetConfig") -> dict[str, Any]:
        fields = super()._dataset_to_fields(ds)
        if getattr(ds, "multimodal_keys", None):
            fields["multimodal_keys"] = ds.multimodal_keys
        return fields

    @staticmethod
    def _validate_custom_model_architecture(
        m: "ModelConfig",
    ) -> "ModelArchitecture":
        if m.architecture is None:
            raise TrainingGymConfigError(
                "SlimeRecipe requires a ModelArchitecture on the attached "
                "ModelConfig. Set `architecture = ModelArchitecture(...)` "
                "on your subclass."
            )
        return m.architecture

    @staticmethod
    def _validate_dataset(ds: "DatasetConfig") -> None:
        """Local preflight for the most common dataset misconfigurations.

        Slime indexes ``data[input_key]`` and ``data[label_key]`` inside a Ray
        actor's ``__init__``; if those are unset or collide, the failure only
        surfaces after image build + Ray bringup. Catch it here instead.
        """
        if not ds.input_key:
            raise TrainingGymConfigError(
                f"{type(ds).__name__}.input_key is unset. Slime requires a "
                "column name (e.g. 'messages' for chat data, 'text' for raw "
                "prompts). Set `input_key = ...` on your DatasetConfig subclass."
            )
        if ds.label_key and ds.label_key == ds.input_key:
            raise TrainingGymConfigError(
                f"{type(ds).__name__}: input_key and label_key are both "
                f"{ds.input_key!r}; they must name distinct columns."
            )

    @staticmethod
    def _model_to_fields(m: "ModelConfig") -> dict[str, Any]:
        arch = SlimeRecipe._validate_custom_model_architecture(m)
        return {
            "hf_checkpoint": m.model_path or m.model_name,
            "num_layers": arch.num_layers,
            "hidden_size": arch.hidden_size,
            "ffn_hidden_size": arch.ffn_hidden_size,
            "num_attention_heads": arch.num_attention_heads,
            "group_query_attention": arch.group_query_attention,
            "num_query_groups": arch.num_query_groups,
            "kv_channels": arch.kv_channels,
            "vocab_size": arch.vocab_size,
            "normalization": arch.normalization,
            "norm_epsilon": arch.norm_epsilon,
            "swiglu": arch.swiglu,
            "disable_bias_linear": arch.disable_bias_linear,
            "qk_layernorm": arch.qk_layernorm,
            "untie_embeddings_and_output_weights": arch.untie_embeddings_and_output_weights,
            **({"num_experts": arch.num_experts} if arch.num_experts else {}),
            **(
                {"moe_ffn_hidden_size": arch.moe_ffn_hidden_size}
                if arch.moe_ffn_hidden_size
                else {}
            ),
            **(
                {
                    "moe_shared_expert_intermediate_size": arch.moe_shared_expert_intermediate_size
                }
                if arch.moe_shared_expert_intermediate_size
                else {}
            ),
            **({"moe_grouped_gemm": True} if arch.moe_grouped_gemm else {}),
            **({"moe_shared_expert_gate": True} if arch.moe_shared_expert_gate else {}),
            **(
                {"moe_router_topk": arch.moe_router_topk}
                if arch.moe_router_topk
                else {}
            ),
            **(
                {"moe_router_score_function": arch.moe_router_score_function}
                if arch.moe_router_score_function
                else {}
            ),
            **(
                {"moe_token_drop_policy": arch.moe_token_drop_policy}
                if arch.moe_token_drop_policy
                else {}
            ),
            **(
                {"moe_router_dtype": arch.moe_router_dtype}
                if arch.moe_router_dtype
                else {}
            ),
            **({"moe_permute_fusion": True} if arch.moe_permute_fusion else {}),
            **(
                {"moe_aux_loss_coeff": arch.moe_aux_loss_coeff}
                if arch.moe_aux_loss_coeff is not None
                else {}
            ),
            **({"spec": arch.megatron_spec} if arch.megatron_spec else {}),
            **({"apply_layernorm_1p": True} if arch.apply_layernorm_1p else {}),
            **({"use_gated_attention": True} if arch.use_gated_attention else {}),
            **({"attention_output_gate": True} if arch.attention_output_gate else {}),
            "use_rotary_position_embeddings": arch.use_rotary_position_embeddings,
            "rotary_base": arch.rotary_base,
            **(
                {"rotary_percent": arch.rotary_percent}
                if arch.rotary_percent != 1.0
                else {}
            ),
        }

    def validate_model_parallelism(self, model: "ModelConfig") -> None:
        validate_num_experts_divisible_by_expert_parallel_size(self, model)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        import dataclasses as _dc

        fields: dict[str, Any] = {}
        for f in _dc.fields(self):
            fields[f.name] = getattr(self, f.name)
        if dataset is not None:
            fields.update(self._dataset_to_fields(dataset))
        if model is not None:
            self.validate_model_parallelism(model)
            if not self.slime_model_script:
                fields.update(self._model_to_fields(model))
        if self.wandb is not None:
            fields.update(self._wandb_to_fields(self.wandb))
        out = {k: v for k, v in fields.items() if k not in _SLIME_SKIP}
        # extra_config is an explicit per-recipe escape hatch and must ALWAYS win
        # over a top-level field's value. slime's --<flag> CLI args override the
        # YAML custom-config, so any key a recipe also sets in extra_config would
        # otherwise be clobbered by the field's CLI flag (e.g. qkv_format="thd"
        # default overriding ASR/VL's extra_config "bshd"). Drop any such CLI flag
        # so the extra_config value stands.
        extra_cfg = fields.get("extra_config")
        if isinstance(extra_cfg, dict):
            for key in extra_cfg:
                out.pop(key, None)
        if "extra_config" in out:
            out["custom_config_path"] = out.pop("extra_config")
        for src, dst in {
            "rollout_function": "rollout_function_path",
            "custom_rollout_log_function": "custom_rollout_log_function_path",
            "custom_eval_rollout_log_function": "custom_eval_rollout_log_function_path",
            "custom_megatron_before_log_prob_hook": "custom_megatron_before_log_prob_hook_path",
            "custom_megatron_before_train_step_hook": "custom_megatron_before_train_step_hook_path",
        }.items():
            if src in _HOOK_WRAPPER_PATHS:
                out[dst] = _HOOK_WRAPPER_PATHS[src]
                continue
            if path := self._path_or_callable_path(fields.get(src)):
                out[dst] = path
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def get_base_recipe(cls, model_config: ModelConfig) -> "SlimeRecipe":
        from modal_training_gym.train_recipes.slime_recipe.glm_4_7 import (
            GLM_4_7_Recipe,
        )
        from modal_training_gym.common.models.megagem_qwen3_4b_sft import (
            MEGAGEM_QWEN3_4B_SFT_MODEL,
        )
        from modal_training_gym.train_recipes.slime_recipe.megagem_qwen3_4b_stage_c import (
            MegaGem_Qwen3_4B_StageC_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_0_6b import (
            Qwen3_0_6b_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_1_7b import (
            Qwen3_1_7b_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_8b import (
            Qwen3_8b_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import (
            Qwen3_4b_Recipe,
        )

        from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import (
            Qwen3_6_35b_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
            Qwen3_ASR_1_7b_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import (
            Qwen3_VL_8b_Recipe,
        )

        if model_config.model_name == "Qwen/Qwen3-VL-8B-Instruct":
            return Qwen3_VL_8b_Recipe()
        if model_config.model_name == "Qwen/Qwen3-ASR-1.7B":
            return Qwen3_ASR_1_7b_Recipe()
        if model_config.model_name == "zai-org/GLM-4.7":
            return GLM_4_7_Recipe()
        if model_config.model_name == MEGAGEM_QWEN3_4B_SFT_MODEL:
            return MegaGem_Qwen3_4B_StageC_Recipe()
        if model_config.model_name == "Qwen/Qwen3-0.6B":
            return Qwen3_0_6b_Recipe()
        if model_config.model_name == "Qwen/Qwen3-1.7B":
            return Qwen3_1_7b_Recipe()
        if model_config.model_name == "Qwen/Qwen3-4B":
            return Qwen3_4b_Recipe()
        if model_config.model_name == "Qwen/Qwen3-8B":
            return Qwen3_8b_Recipe()
        if model_config.model_name == "Qwen/Qwen3.6-35B-A3B":
            return Qwen3_6_35b_Recipe()
        raise TrainingGymConfigError(
            f"no base slime recipe for model {model_config.model_name!r}"
        )
