import re
from collections.abc import Callable
from dataclasses import field
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

import modal
from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.metric_mirror import DashboardMetricConfig
from modal_training_gym.common.metrics import MetricConfig
from modal_training_gym.common.models import (
    ModelArchitecture,
    ModelConfig,
)
from modal_training_gym.train_recipes.base import (
    # Re-exported for backwards compatibility (e.g. frameworks/slime/launcher.py
    # imports the volume paths from this module).
    CHECKPOINTS_PATH as CHECKPOINTS_PATH,
)
from modal_training_gym.train_recipes.base import (
    DATA_PATH as DATA_PATH,
)
from modal_training_gym.train_recipes.base import (
    HF_CACHE_PATH as HF_CACHE_PATH,
)
from modal_training_gym.train_recipes.base import (
    JSON_CONFIG_FIELDS as JSON_CONFIG_FIELDS,
)
from modal_training_gym.train_recipes.base import (
    BaseTrainRecipe,
    _apply_loss_type_fields,
)
from modal_training_gym.train_recipes.gpu_allocation import (
    resolve_gpu_allocation,
    validate_megatron_actor_parallelism,
    validate_multi_node_gpu_count,
    validate_num_experts_divisible_by_expert_parallel_size,
)

# ── Types ─────────────────────────────────────────────────────────────────────

_SLIME_SKIP = {
    "environment",
    "async_mode",
    "metrics",
    "name",
    "app_tags",
    "capture_trace",
    "trace_sample_limit",
    "image_overlay",
    "local_slime",
    "slime_git_repository",
    "slime_git_revision",
    "data_volume_name",
    "memory",
    "cpu",
    "cloud",
    "region",
    "checkpoint",
    "custom_rm_function",
    "custom_generate_function",
    "custom_reward_post_process_function",
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
    "max_retries",
    "substep_timing",
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
    """Slime training and Modal resource settings.

    Args:

        name:
            Modal app title. The launcher derives it from the recipe class when
            empty.
        app_tags:
            Extra tags merged into the Modal app metadata for dashboard
            auto-discovery.

        gpu_type:
            Modal GPU type for every node.
        colocate:
            Share GPUs between the trainer and rollout engines.
        actor_num_nodes:
            Megatron actor nodes.
        actor_num_gpus_per_node:
            GPUs per actor node.
        rollout_num_gpus:
            Total GPUs for rollout engines when disaggregated; ``None`` lets the
            allocation resolver size it.
        rollout_num_gpus_per_engine:
            GPUs and tensor-parallel size per SGLang engine.
        use_critic:
            Train a separate critic model for PPO. GRPO does not use one.
        critic_num_nodes:
            Nodes for the critic when ``use_critic`` is set.
        critic_num_gpus_per_node:
            GPUs per critic node.
        tensor_model_parallel_size:
            Megatron tensor-parallel size for the actor.
        sequence_parallel:
            Megatron sequence parallelism. Requires tensor parallelism greater than one.

        num_rollout:
            Training and rollout steps for the run.
        start_rollout_id:
            Rollout step to start counting from. ``None`` continues from the
            iteration stored in ``load``.
        rollout_batch_size:
            Prompts sampled per rollout step; each prompt is expanded into a
            group of sampled responses.
        n_samples_per_prompt:
            Responses sampled per prompt as one GRPO group.
        rollout_max_response_len:
            Max generated tokens per sample.
        rollout_temperature:
            Sampling temperature for rollout generation.
        rollout_shuffle:
            Shuffle the prompt dataset between epochs.
        rollout_top_p:
            Nucleus-sampling top-p for rollout generation.
        rollout_stop_token_ids:
            Extra token ids that terminate generation.

        use_fault_tolerance:
            Enable slime's fault tolerance so the run can recover from worker
            failures.
        rollout_health_check_interval:
            Interval in seconds between rollout engine /health_generate checks
            during generate/eval.
        rollout_health_check_timeout:
            Timeout in seconds to wait for a rollout engine /health_generate
            response before killing it.
        rollout_health_check_first_wait:
            Initial delay in seconds before health checks. DeepGEMM compilation may
            require a longer delay.

        sglang_mem_fraction_static:
            Fraction of GPU memory sglang reserves for weights + KV cache.
        sglang_enable_dp_attention:
            Enable data-parallel attention across engine ranks.
        sglang_dp_size:
            Data-parallel size for the engines.
        sglang_ep_size:
            Expert-parallel size for MoE models.
        sglang_enable_dp_lm_head:
            Data-parallel LM head paired with DP attention.
        sglang_disable_custom_all_reduce:
            Fall back to NCCL all-reduce instead of sglang's custom kernel.
        sglang_cuda_graph_bs:
            Batch sizes to capture CUDA graphs for.
        sglang_max_running_requests:
            Cap on concurrent in-flight requests per engine.
        sglang_tool_call_parser:
            Tool-call output parser.
        sglang_reasoning_parser:
            Parser for reasoning/thinking output.
        sglang_request_params:
            Additional parameters for SGLang generation requests.

        advantage_estimator:
            Advantage estimator.
        eps_clip:
            PPO clip lower bound.
        eps_clip_high:
            Upper PPO clip bound for asymmetric DAPO clipping.
        use_kl_loss:
            Add a per-token KL loss term against the reference model.
        kl_loss_type:
            KL formulation.
        kl_loss_coef:
            Coefficient of the KL loss term.
        kl_coef:
            KL penalty coefficient applied in the reward.
        entropy_coef:
            Entropy bonus coefficient.
        calculate_per_token_loss:
            Average the loss over tokens instead of over samples.
        loss_type:
            ``"policy_loss"`` for RL or ``"sft_loss"`` for supervised
            fine-tuning, which overrides conflicting RL settings.
        loss_mask_type:
            Chat-template loss mask for SFT. Slime choices are ``qwen``,
            ``qwen3``, ``qwen3_5``, and ``distill_qwen`` (upstream default
            ``qwen``). Emitted only under ``sft_loss`` when not the default.
        num_epoch:
            Passes over the dataset. Replaces ``num_rollout`` when set.

        over_sampling_batch_size:
            Extra DAPO prompts sampled to replace filtered groups.
        dynamic_sampling_filter_path:
            Import path of the predicate that selects sample groups.
        balance_data:
            Rebalance kept samples across data-parallel ranks.

        global_batch_size:
            Training samples per optim step.
        num_steps_per_rollout:
            Optimizer steps per rollout. ``None`` omits the flag.
        lr:
            Learning rate.
        lr_decay_style:
            Learning-rate schedule.
        weight_decay:
            Weight decay.
        adam_beta1:
            Adam beta1.
        adam_beta2:
            Adam beta2.
        optimizer:
            Optimizer name.
        use_distributed_optimizer:
            Shard optimizer state across data-parallel ranks with Megatron's
            distributed optimizer.
        optimizer_cpu_offload:
            Keep optimizer state on CPU to reduce GPU memory use at the cost of
            slower steps.
        overlap_cpu_optimizer_d2h_h2d:
            Overlap the offloaded optimizer's device↔host copies with compute.
        use_precision_aware_optimizer:
            Use Megatron's precision-aware optimizer with lower-precision state.

        attention_dropout:
            Attention dropout probability.
        hidden_dropout:
            Hidden-layer dropout probability.
        attention_softmax_in_fp32:
            Compute attention softmax in fp32.
        accumulate_allreduce_grads_in_fp32:
            Accumulate and all-reduce gradients in fp32.
        attention_backend:
            Megatron attention kernel backend.
        recompute_granularity:
            Activation recomputation granularity: ``"full"`` or ``"selective"``.
        recompute_method:
            Recomputation method: ``"uniform"`` or ``"block"``.
        recompute_num_layers:
            Layers per recomputation chunk.
        qkv_format:
            QKV layout for the Megatron backend, emitted as ``--qkv-format``.

        use_dynamic_batch_size:
            Pack variable-length samples into micro-batches up to
            ``max_tokens_per_gpu`` instead of a fixed micro batch size.
        max_tokens_per_gpu:
            Token budget per GPU per micro-batch when dynamic batching is on.

        rm_type:
            Built-in reward function name. Leave unset for a custom reward.

        custom_rm_function:
            Reward callable shipped by value to the containers and registered as
            slime's ``--custom-rm-path``.
        custom_generate_function:
            Custom slime generation step shipped by value.
        custom_reward_post_process_function:
            Function applied to rewards after generation and shipped by value.
        rollout_function:
            Custom rollout loop passed through ``--rollout-function-path``.
        custom_rollout_log_function:
            Function called with each rollout's data after dashboard and phase
            reporting.
        custom_eval_rollout_log_function:
            Function called with each evaluation rollout's data.
        custom_megatron_before_log_prob_hook:
            Hook run in the Megatron trainer before log-prob computation.
        custom_megatron_before_train_step_hook:
            Hook run in the Megatron trainer before each train step.

        update_weight_mode:
            Weight synchronization mode. ``"full"`` sends all weights. ``"delta"``
            sends byte-level changes from a CPU snapshot.
        update_weight_transport:
            ``"nccl"`` or ``"disk"``; disk requires trainer and rollout engines
            to share a filesystem.
        update_weight_encoding:
            Encoding for delta payloads.
        update_weight_disk_dir:
            Shared directory used by the disk transport.

        save:
            Checkpoint output directory on the mounted ``/checkpoints`` volume.
        save_interval:
            Save a checkpoint every N rollout steps. Defaults to num_rollout, so a
            run checkpoints once at the end.
        load:
            Checkpoint directory to resume from; empty starts from the converted
            HF weights.
        ref_load:
            Checkpoint read by the reference model for KL terms.
        no_save_optim:
            Omit optimizer state from checkpoints. The resulting checkpoints cannot
            resume the optimizer exactly.
        no_load_optim:
            Skip loading optimizer and scheduler state from ``load``.
        megatron_to_hf_mode:
            Export mode for saved Megatron checkpoints. An empty value disables
            export.
        freeze_params_name_list:
            Parameter-name patterns matched with ``re.search`` to select frozen
            weights.
        source_hf_checkpoint:
            Source checkpoint when it differs from the model's own.
        megatron_conversion_hf_checkpoint:
            HF checkpoint used for the HF→Megatron conversion step instead of the
            training model's own weights.

        eval_interval:
            Run eval every N rollout steps; ``None`` disables eval.
        n_samples_per_eval_prompt:
            Responses sampled per eval prompt.
        eval_max_response_len:
            Max generated tokens per eval sample.
        eval_top_p:
            Nucleus-sampling top-p for eval generation.
        eval_config:
            Evaluation defaults and datasets written to ``--eval-config`` as YAML.

        environment:
            Training-container environment variables such as Megatron
            ``PYTHONPATH`` and NCCL settings.
        async_mode:
            Overlap rollout generation and training with slime's one-step off-policy
            ``train_async.py``.
        metrics:
            Metric tracker settings; expands to slime's W&B-compatible flags.
            Defaults to the dashboard-only tracker; ``None`` disables metric
            logging entirely.
        image_overlay:
            Function that modifies the Modal image.
        local_slime:
            Local slime checkout mounted over the image copy without rebuilding it.
        slime_git_repository:
            Public HTTPS Git repository to overlay onto the image's slime checkout.
            Must be paired with ``slime_git_revision`` and is intended for
            reproducible fork-backed runs. The selected source must remain compatible
            with Training Gym's build-time Slime patches.
        slime_git_revision:
            Full 40-character commit SHA fetched from ``slime_git_repository``.
            Branches and tags are rejected because they can move between runs.
        data_volume_name:
            Existing Modal data volume to mount at ``/data``. When unset, the
            launcher derives a volume name from the concrete recipe class.
        slime_model_script:
            Slime script that defines ``MODEL_ARGS`` in place of the attached
            ``ModelConfig`` architecture.
        memory:
            Modal Function memory request/limit in MiB.
        cpu:
            Modal Function CPU request/limit in cores per container.
        cloud:
            Modal cloud provider to pin the cluster to.
        region:
            Modal region to pin the cluster to.
        patch_files:
            Local patch scripts applied to slime and Megatron sources.
        image_run_commands:
            Extra shell commands run while building the image.
        image_env:
            Extra env vars baked into the image.
        train_function_kwargs:
            Additional Modal Function keyword arguments for the training function.
        max_retries:
            Modal retries for the training function. Each retry resumes from
            the last checkpoint.
        capture_trace:
            Attach sampled per-request execution traces to recorded rollouts.
        trace_sample_limit:
            Maximum traced samples per rollout when ``capture_trace`` is enabled.

        extra_config:
            Custom configuration written to YAML and passed as
            ``--custom-config-path``. Keys become attributes on slime's parsed
            args and always override same-named recipe fields.
        sglang_config:
            SGLang engine settings written to ``--sglang-config`` as YAML.
        apply_chat_template_kwargs:
            Keyword arguments for tokenizer ``apply_chat_template``, passed as JSON.
        train_env_vars:
            Env vars for the training processes, passed as inline JSON.
        multimodal_keys:
            Multimodal dataset columns passed as JSON.
        sglang_cuda_graph_backend_prefill:
            SGLang CUDA-graph backend used during prefill.
        substep_timing:
            Record per-substep timings for the dashboard. Defaults to ``auto``,
            which enables substep time reporting.
    """

    # ── App identity ─────────────────────────────────────────────────────────
    name: str = ""
    app_tags: dict = field(default_factory=dict)

    # ── Cluster ─────────────────────────────────
    gpu_type: str = "H100"
    colocate: bool = True
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 1
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int = 1
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None

    # ── Parallelism ─────────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False

    # ── Rollout and sampling ─────────────────────────────────────────────────
    num_rollout: int = 1
    start_rollout_id: int | None = None
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 2
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    rollout_shuffle: bool = True
    rollout_top_p: float = 1.0
    rollout_stop_token_ids: list[int] | None = None

    # ── Fault Tolerance and Health Checks ───────────────────────────────────
    use_fault_tolerance: bool = True
    rollout_health_check_interval: int = 30
    rollout_health_check_timeout: int = 30
    rollout_health_check_first_wait: int = 300

    # ── SGLang rollout engine ──────────────────────────────────────────────
    sglang_mem_fraction_static: float = 0.75
    sglang_enable_dp_attention: bool = False
    sglang_dp_size: int | None = None
    sglang_ep_size: int | None = None
    sglang_enable_dp_lm_head: bool = False
    sglang_disable_custom_all_reduce: bool = False
    sglang_cuda_graph_bs: list[int] | None = None
    sglang_cuda_graph_backend_prefill: str | None = None
    sglang_max_running_requests: int | None = None
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None
    sglang_request_params: dict | None = None

    # ── RL algorithm ────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = False
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    calculate_per_token_loss: bool = False
    loss_type: Literal["policy_loss", "sft_loss"] = "policy_loss"
    loss_mask_type: Literal["qwen", "qwen3", "qwen3_5", "distill_qwen"] = "qwen"

    # ── Dynamic sampling (DAPO) ────────────────────────────────────────────
    over_sampling_batch_size: int | None = None
    dynamic_sampling_filter_path: str | None = None
    balance_data: bool = False

    # ── Training ────────────────────────────────────────────────────────────
    global_batch_size: int = 4
    num_epoch: int | None = None
    num_steps_per_rollout: int | None = None
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"
    use_distributed_optimizer: bool = False
    optimizer_cpu_offload: bool = False
    overlap_cpu_optimizer_d2h_h2d: bool = False
    use_precision_aware_optimizer: bool = False

    # ── Memory and precision ────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_backend: str | None = None
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    qkv_format: str = "thd"

    # ── Dynamic batching ────────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216

    # ── Reward model ─────────────────────────────────────────────────────────
    rm_type: str | None = None

    # -- Slime customization flags ───────────────────────────────────────────
    custom_rm_function: Callable | None = None
    custom_generate_function: Callable | None = None
    custom_reward_post_process_function: Callable | None = None
    rollout_function: Callable | str | None = None
    custom_rollout_log_function: Callable | str | None = None
    custom_eval_rollout_log_function: Callable | str | None = None
    custom_megatron_before_log_prob_hook: Callable | str | None = None
    custom_megatron_before_train_step_hook: Callable | str | None = None

    # ── Weight sync ──────────
    update_weight_mode: str = "full"
    update_weight_transport: str = "nccl"
    update_weight_encoding: str = "indices"
    update_weight_disk_dir: str = ""

    # ── Checkpointing ───────────────────────────────────────────
    save: str | None = str(CHECKPOINTS_PATH)
    save_interval: int | None = None
    load: str = ""
    ref_load: str = ""
    no_save_optim: bool = False
    no_load_optim: bool = False
    megatron_to_hf_mode: str = ""
    freeze_params_name_list: list[str] | None = None
    source_hf_checkpoint: str | None = None
    megatron_conversion_hf_checkpoint: str | None = None

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 2
    eval_max_response_len: int = 4096
    eval_top_p: float = 1.0
    eval_config: dict | None = None

    # ── Launcher instructions ─────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    metrics: MetricConfig | None = field(default_factory=DashboardMetricConfig)
    image_overlay: Callable[[modal.Image], modal.Image] | None = None
    local_slime: str | None = None
    slime_git_repository: str | None = None
    slime_git_revision: str | None = None
    data_volume_name: str | None = None
    slime_model_script: str = ""
    memory: int | tuple[int, int] | None = None
    cpu: float | tuple[float, float] | None = None
    cloud: str | None = None
    region: str | None = None
    patch_files: list[str] = field(default_factory=list)
    image_run_commands: list[str] = field(default_factory=list)
    image_env: dict[str, str] = field(default_factory=dict)
    train_function_kwargs: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    substep_timing: Literal["auto", "off"] = "auto"

    # ── Per-sample execution tracing (dashboard timeline) ───────────────────
    capture_trace: bool = False
    trace_sample_limit: int = 16

    # ── SGLang / config overrides ───────────────────────────────────────────
    extra_config: dict | None = None
    sglang_config: dict | None = None
    apply_chat_template_kwargs: dict | str = ""
    train_env_vars: dict | str | None = None
    multimodal_keys: dict | str | None = None

    # ── Validators ───────────────────────────────────────────────────────────

    _SKIP_FIELDS: ClassVar[frozenset[str]] = frozenset(_SLIME_SKIP)

    @model_validator(mode="after")
    def _validate_slime_source_overlay(self) -> "SlimeRecipe":
        repository = self.slime_git_repository
        revision = self.slime_git_revision
        if bool(repository) != bool(revision):
            raise TrainingGymConfigError(
                "slime_git_repository and slime_git_revision must be set together"
            )
        if self.local_slime and repository:
            raise TrainingGymConfigError(
                "local_slime and slime_git_repository are mutually exclusive"
            )
        if repository:
            parsed = urlparse(repository)
            if parsed.scheme != "https" or not parsed.netloc:
                raise TrainingGymConfigError(
                    "slime_git_repository must be a public HTTPS URL"
                )
            if parsed.username or parsed.password:
                raise TrainingGymConfigError(
                    "slime_git_repository must not contain credentials"
                )
            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
                raise TrainingGymConfigError(
                    "slime_git_revision must be a full 40-character commit SHA"
                )
            object.__setattr__(self, "slime_git_revision", revision.lower())
        return self

    @model_validator(mode="after")
    def _resolve_callable_paths(self) -> "SlimeRecipe":
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
        validate_multi_node_gpu_count(resolve_gpu_allocation(self), self.gpu_type)
        validate_megatron_actor_parallelism(self)
        return self

    # ── Container → slime flag converters ────────────────────────────────────

    @classmethod
    def _dataset_to_fields(
        cls,
        ds: "DatasetConfig",
        eval_ds: "DatasetConfig | None" = None,
        *,
        dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
    ) -> dict[str, Any]:
        fields = super()._dataset_to_fields(
            ds,
            eval_ds,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
        )
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

    @classmethod
    def _validate_datasets(
        cls,
        ds: "DatasetConfig",
        eval_ds: "DatasetConfig | None" = None,
        *,
        loss_type: str = "policy_loss",
    ) -> None:
        """Local preflight for the most common dataset misconfigurations.

        Slime indexes ``data[input_key]`` and ``data[label_key]`` inside a Ray
        actor's ``__init__``; if those are unset or collide, the failure only
        surfaces after image build + Ray bringup. Catch it here instead.
        """
        super()._validate_datasets(ds, eval_ds, loss_type=loss_type)
        for dataset in (ds, eval_ds):
            if dataset is None:
                continue
            input_key = dataset.input_key()
            label_key = dataset.label_key()
            if not input_key:
                raise TrainingGymConfigError(
                    f"{type(dataset).__name__}.input_key() is unset. Slime requires a "
                    "column name (e.g. 'messages' for chat data, 'text' for raw "
                    "prompts). Implement `input_key()` on your DatasetConfig subclass."
                )
            if label_key and label_key == input_key:
                raise TrainingGymConfigError(
                    f"{type(dataset).__name__}: input_key() and label_key() are both "
                    f"{input_key!r}; they must name distinct columns."
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
        eval_dataset: "DatasetConfig | None" = None,
        dataset_path: str | None = None,
        eval_dataset_path: str | None = None,
        model: "ModelConfig | None" = None,
    ) -> dict[str, Any]:
        fields = self._field_values()
        if fields["save_interval"] is None and fields["save"] is not None:
            fields["save_interval"] = self._escape_hatch_values().get(
                "num_rollout", self.num_rollout
            )
        if (
            self.colocate
            and self.loss_type != "sft_loss"
            and fields["sglang_cuda_graph_backend_prefill"] is None
            and "sglang_cuda_graph_backend_prefill" not in self._escape_hatch_keys()
        ):
            fields["sglang_cuda_graph_backend_prefill"] = "disabled"
        if dataset is not None:
            fields.update(
                self._dataset_to_fields(
                    dataset,
                    eval_dataset,
                    dataset_path=dataset_path,
                    eval_dataset_path=eval_dataset_path,
                )
            )
        _apply_loss_type_fields(
            fields,
            sft_rollout_function="slime.rollout.sft_rollout.generate_rollout",
        )
        if model is not None:
            self.validate_model_parallelism(model)
            if not self.slime_model_script:
                fields.update(self._model_to_fields(model))
        if self.metrics is not None:
            fields.update(self._metrics_to_fields(self.metrics))
        out = self._emit_fields(fields)
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

    @property
    def train_async(self) -> bool:
        return self.async_mode and self.loss_type != "sft_loss"

    @classmethod
    def get_base_recipe(cls, model_config: ModelConfig) -> "SlimeRecipe":
        from modal_training_gym.train_recipes.slime_recipe.glm_4_7 import (
            GLM_4_7_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_0_6b import (
            Qwen3_0_6B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_1_7b import (
            Qwen3_1_7B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import (
            Qwen3_4B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_5_0_8b import (
            Qwen3_5_0_8B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_5_2b import (
            Qwen3_5_2B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_5_4b import (
            Qwen3_5_4B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_5_9b import (
            Qwen3_5_9B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b import (
            Qwen3_6_27B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_6_35b import (
            Qwen3_6_35B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_8_27b import (
            Qwen3_8_27B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_8b import (
            Qwen3_8B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_asr_1_7b import (
            Qwen3_ASR_1_7B_Recipe,
        )
        from modal_training_gym.train_recipes.slime_recipe.qwen3_vl_8b import (
            Qwen3_VL_8B_Recipe,
        )

        if model_config.model_name == "Qwen/Qwen3-VL-8B-Instruct":
            return Qwen3_VL_8B_Recipe()
        if model_config.model_name == "Qwen/Qwen3-ASR-1.7B":
            return Qwen3_ASR_1_7B_Recipe()
        if model_config.model_name == "zai-org/GLM-4.7":
            return GLM_4_7_Recipe()
        if model_config.model_name == "Qwen/Qwen3-0.6B":
            return Qwen3_0_6B_Recipe()
        if model_config.model_name == "Qwen/Qwen3-1.7B":
            return Qwen3_1_7B_Recipe()
        if model_config.model_name == "Qwen/Qwen3-4B":
            return Qwen3_4B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.5-0.8B":
            return Qwen3_5_0_8B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.5-2B":
            return Qwen3_5_2B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.5-4B":
            return Qwen3_5_4B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.5-9B":
            return Qwen3_5_9B_Recipe()
        if model_config.model_name == "Qwen/Qwen3-8B":
            return Qwen3_8B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.6-35B-A3B":
            return Qwen3_6_35B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.6-27B":
            return Qwen3_6_27B_Recipe()
        if model_config.model_name == "Qwen/Qwen3.8-27B":
            return Qwen3_8_27B_Recipe()
        raise TrainingGymConfigError(
            f"no base slime recipe for model {model_config.model_name!r}"
        )
