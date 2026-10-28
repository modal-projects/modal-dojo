"""Beam model and LoRA defaults from radixark/miles#4149."""

from dataclasses import MISSING, field, fields
import json
from math import lcm
from pathlib import Path
from typing import Callable, ClassVar

from pydantic import ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.models.beam import Beam_1_0_501B_A23B_LoRA
from modal_dojo.train_recipes.base import MicroBatchSize
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


async def beam_reward(args, sample, **kwargs):
    # Import in the pinned remote runtime; preserve Dojo's reward/timing wrapper.
    from examples.beam.reward import reward

    return await reward(args, sample, **kwargs)


def _image_commands() -> list[str]:
    # Match Dockerfile.beam without upgrading torch or its prebuilt extensions.
    return [
        "cd /sgl-workspace/sglang && SGLANG_BUILD_RUST_EXTS=none "
        'pip install -e "python[all]" --no-deps',
        "pip install -e /root/miles --no-deps --no-build-isolation",
    ]


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _BeamRecipe(MilesRecipe):
    _BEAM_ATTENTION_BACKENDS: ClassVar[tuple[str, ...]] = ("unfused",)

    docker_image: str = (
        "radixark/miles:v0.1.2@sha256:"
        "c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
    )
    miles_git_ref: str | None = "819cfd23ab74de9975931597894500632dd708fc"
    sglang_git_ref: str | None = "9ac1c65bec3dc832766d03970e545da5bd6fcb6f"
    image_run_commands: list[str] = field(default_factory=_image_commands)
    miles_model_name: str = "beam"
    model_name: str = "beam"

    gpu_type: str = "H200"
    colocate: bool = False
    async_mode: bool = True
    actor_num_nodes: int = 2
    rollout_num_gpus: int | None = 16
    expert_tensor_parallel_size: int = 8
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus_per_engine: int = 16
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    cpu: float | tuple[float, float] | None = 32
    memory: tuple[int, int] = (256 * 1024, 2 * 1024 * 1024)

    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    rollout_max_prompt_len: int = 1024
    rollout_max_context_len: int = 5120
    rollout_top_k: int | None = -1
    rollout_seed: int = 1234
    seed: int = 1234
    balance_data: bool = True
    custom_rm_function: Callable | None = beam_reward
    reward_key: str | None = "score"
    eval_reward_key: str | None = "acc"
    n_samples_per_eval_prompt: int = 8
    # Evaluate when an eval dataset/config is supplied; never reuse training data.
    eval_interval: int | None = 1

    # Beam's custom attention requires padded bshd, not dynamic thd packing.
    use_dynamic_batch_size: bool = False
    micro_batch_size: MicroBatchSize = 1
    qkv_format: str = "bshd"
    attention_backend: str = "unfused"
    seq_length: int = 5120
    data_pad_size_multiplier: int = 1
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_bias_update_rate: float = 0.0
    moe_aux_loss_coeff: float = 0.0
    bf16: bool = True
    data_parallel_random_init: bool = True
    accumulate_allreduce_grads_in_fp32: bool = False
    grad_reduce_in_bf16: bool = True

    eps_clip_c: float = 10.0
    calculate_per_token_loss: bool = True
    use_tis: bool = True
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    clip_grad: float = 1.0
    use_distributed_optimizer: bool = True
    no_gradient_accumulation_fusion: bool = True
    ckpt_format: str = "torch_dist"
    use_checkpoint_opt_param_scheduler: bool = True

    sglang_dtype: str = "bfloat16"
    # Match pinned SGLang Beam defaults; Triton decode drops one SWA token.
    sglang_attention_backend: str = "trtllm_mha"
    sglang_tool_call_parser: str | None = "beam"
    sglang_reasoning_parser: str | None = "beam"
    sglang_moe_runner_backend: str | None = "triton"
    sglang_moe_a2a_backend: str = "none"
    sglang_ep_size: int | None = 1
    sglang_pp_size: int = 1
    sglang_context_length: int = 5120
    sglang_chunked_prefill_size: int = 8192
    sglang_mem_fraction_static: float = 0.85
    update_weight_transfer_mode: str = "broadcast"
    update_weight_buffer_size: int | None = 268435456
    update_weights_interval: int = 1
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/miles:/root/Megatron-LM",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_CUMEM_ENABLE": "1",
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_MNNVL_ENABLE": "0",
            "NCCL_GRAPH_REGISTER": "0",
            "NCCL_ALGO": "allreduce:tree",
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "8",
            "MILES_USE_LEGACY_ROLLOUT_V1": "0",
            # Broadcast weight sync does not need the Mooncake RDMA reinstall.
            "MILES_REINSTALL_RDMA": "0",
            "MILES_CELL_READY_TIMEOUT_SECONDS": "10800",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RAY_USAGE_STATS_ENABLED": "0",
            "DO_NOT_TRACK": "1",
        }
    )

    @model_validator(mode="before")
    @classmethod
    def _preserve_beam_defaults(cls, values):
        # Merge before base validators inspect patches and the effective env.
        if isinstance(values, ArgsKwargs):
            positional = dict(zip((f.name for f in fields(cls)), values.args))
            options = {**positional, **(values.kwargs or {})}
        elif isinstance(values, dict):
            options = dict(values)
        else:
            return values
        defaults = {f.name: f for f in fields(cls)}
        for name in (
            "patch_files",
            "image_run_commands",
            "environment",
            "train_env_vars",
        ):
            if name not in options:
                continue
            definition = defaults[name]
            required = (
                definition.default_factory()
                if definition.default_factory is not MISSING
                else definition.default
            )
            supplied = options[name]
            if name in ("environment", "train_env_vars"):
                if isinstance(supplied, str):
                    supplied = json.loads(supplied)
                options[name] = {**(required or {}), **(supplied or {})}
            else:
                if name == "patch_files":
                    protected = {Path(path).name: path for path in required}
                    for path in supplied or []:
                        original = protected.get(Path(path).name)
                        if (
                            original
                            and Path(original).resolve() != Path(path).resolve()
                        ):
                            raise ValueError(
                                f"Cannot replace required Beam patch {Path(path).name}"
                            )
                options[name] = list(dict.fromkeys([*required, *(supplied or [])]))
        extra = dict(options.get("extra_config") or {})
        if "train_env_vars" in extra:
            supplied = extra["train_env_vars"]
            if isinstance(supplied, str):
                supplied = json.loads(supplied)
            definition = defaults["train_env_vars"]
            required = (
                definition.default_factory()
                if definition.default_factory is not MISSING
                else {}
            )
            extra["train_env_vars"] = {**(required or {}), **(supplied or {})}
            options["extra_config"] = extra
        # A native reward path takes precedence over the preset's math callable.
        custom_reward = extra.get("custom_rm_path") or (
            "custom_rm_function" in options
            and options["custom_rm_function"] is not beam_reward
        )
        if extra.get("custom_rm_path"):
            options["custom_rm_function"] = None
        if custom_reward:
            options.setdefault("reward_key", None)
            options.setdefault("eval_reward_key", None)
        # Explicit iteration warmup remains available without a conflicting fraction.
        if (
            options.get("lr_warmup_iters") is not None
            or extra.get("lr_warmup_iters") is not None
        ):
            options.setdefault("lr_warmup_fraction", None)
        return options

    def _fields(self, **kwargs):
        result = super()._fields(**kwargs)
        extra = self._escape_hatch_values()
        if not (
            kwargs.get("eval_dataset") is not None
            or self.eval_config
            or extra.get("eval_config")
            or extra.get("eval_prompt_data")
        ):
            result.pop("eval_interval", None)
        return result

    @model_validator(mode="after")
    def _check_beam_attention(self) -> "_BeamRecipe":
        overrides = self._escape_hatch_values()

        def effective(name):
            return overrides.get(name, getattr(self, name))

        if effective("use_dynamic_batch_size") or effective("qkv_format") != "bshd":
            raise ValueError("Beam requires fixed microbatches and qkv_format=bshd")
        if effective("attention_backend") not in self._BEAM_ATTENTION_BACKENDS:
            raise ValueError(
                f"This Beam preset requires attention_backend in {self._BEAM_ATTENTION_BACKENDS}"
            )
        if effective("sglang_attention_backend") != "trtllm_mha":
            raise ValueError("Pinned Beam requires sglang_attention_backend=trtllm_mha")
        if self.async_mode and self.colocate:
            raise ValueError("Beam async training requires separate rollout GPUs")
        cp = effective("context_parallel_size")
        if (
            effective("qkv_format") == "bshd"
            and cp > 1
            and effective("sequence_parallel")
        ):
            tp = effective("tensor_model_parallel_size")
            multiplier = effective("data_pad_size_multiplier")
            # CP uses two zigzag chunks per rank; each local sequence must fit TP.
            if tp < 1 or multiplier < 1 or (tp * multiplier) % (cp * lcm(tp, 2)):
                raise ValueError(
                    "Beam bshd padding must preserve CP and sequence-parallel "
                    "alignment; increase data_pad_size_multiplier"
                )
        return self


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class _BeamLoRABase(_BeamRecipe):
    """Upstream rank-16 async GRPO on 16 trainer + 16 rollout H200 GPUs."""

    model_config_class: ClassVar[type[ModelConfig]] = Beam_1_0_501B_A23B_LoRA
    expert_model_parallel_size: int = 16
    expert_tensor_parallel_size: int = 1
    lora_rank: int | None = 16
    lora_alpha: int | None = 32
    lora_dropout: float | None = 0.0
    # Native resume requires the checkpoint patch supplied by the 256k recipe.
    max_retries: int = 0
    # Leave target_modules unset: the pinned Beam registry includes the output gate.
    lr: float = 1e-4
    min_lr: float = 1e-5
    lr_decay_style: str = "cosine"
    # Let Miles derive decay from the actual training horizon (including epochs).
    lr_warmup_iters: int | None = None
    lr_warmup_fraction: float | None = 0.025
    lr_decay_iters: int | None = None
    use_rollout_routing_replay: bool = True
    use_miles_router: bool = True
    sglang_max_running_requests: int | None = 256
    sglang_cuda_graph_max_bs_decode: int = 256

    @model_validator(mode="after")
    def _check_beam_lora(self) -> "_BeamLoRABase":
        if self.max_retries != 0 and not any(
            Path(path).name == "patch_beam_checkpoint.py" for path in self.patch_files
        ):
            raise ValueError(
                "Beam LoRA requires max_retries=0 until adapter resume is supported"
            )
        if self.expert_tensor_parallel_size != 1:
            raise ValueError("Beam expert LoRA requires expert_tensor_parallel_size=1")
        if self.lora_rank is None or self.lora_rank <= 0:
            raise ValueError("Beam LoRA requires a positive lora_rank")
        return self
