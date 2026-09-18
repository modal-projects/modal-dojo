"""Framework-agnostic Modal launcher utilities shared by slime and miles.

Both frameworks' ``modal_helpers/utils.py`` are thin wrappers over these
implementations; they pass framework-specific parametrization (extended MoE
conversion args, the model-script attribute name, single-rank MTP handling) so
the shared logic stays in one place.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from os import PathLike
from typing import Any

# (attr_name_on_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
]

# Architecture fields emitted as CLI flags for every framework.
_ARCH_VALUE_FIELDS = [
    ("num_layers", "num-layers"),
    ("hidden_size", "hidden-size"),
    ("ffn_hidden_size", "ffn-hidden-size"),
    ("num_attention_heads", "num-attention-heads"),
    ("num_query_groups", "num-query-groups"),
    ("kv_channels", "kv-channels"),
    ("vocab_size", "vocab-size"),
    ("norm_epsilon", "norm-epsilon"),
    ("rotary_base", "rotary-base"),
]

_REDACTED_ENV_VALUE = "[redacted]"


def _is_sensitive_env_name(name: object) -> bool:
    """Return whether an environment-variable-like name carries credentials."""
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.endswith("_token_id") or normalized.endswith("_token_ids"):
        return False
    return any(
        marker in normalized
        for marker in (
            "api_key",
            "access_key",
            "private_key",
            "secret",
            "password",
            "passwd",
            "token",
            "credential",
            "authorization",
            "cookie",
        )
    )


def redact_runtime_env(value: Any) -> Any:
    """Return a recursively redacted copy suitable for launcher diagnostics."""
    if isinstance(value, Mapping):
        return {
            key: (
                _REDACTED_ENV_VALUE
                if _is_sensitive_env_name(key) and item not in (None, "")
                else redact_runtime_env(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_runtime_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_runtime_env(item) for item in value)
    return value


def is_local_checkpoint_ref(ref: str | PathLike) -> bool:
    """Return True when a checkpoint ref is already a mounted local path."""
    return str(ref).startswith("/")


def resolve_checkpoint_ref(
    ref: str | PathLike,
    *,
    local_files_only: bool = True,
) -> str:
    """Resolve an absolute path or Hugging Face repo ID to a local path."""
    ref_str = str(ref)
    if is_local_checkpoint_ref(ref_str):
        return ref_str

    from huggingface_hub import snapshot_download

    return snapshot_download(ref_str, local_files_only=local_files_only)


def _append_common_arch_args(extra_args: list[str], arch: Any) -> None:
    for attr, flag in _ARCH_VALUE_FIELDS:
        val = getattr(arch, attr, 0)
        if val:
            extra_args.append(f"--{flag} {val}")
    if arch.group_query_attention:
        extra_args.append("--group-query-attention")
    if arch.swiglu:
        extra_args.append("--swiglu")
    if arch.disable_bias_linear:
        extra_args.append("--disable-bias-linear")
    if arch.qk_layernorm:
        extra_args.append("--qk-layernorm")
    if arch.untie_embeddings_and_output_weights:
        extra_args.append("--untie-embeddings-and-output-weights")
    if arch.normalization and arch.normalization != "LayerNorm":
        extra_args.append(f"--normalization {arch.normalization}")


def _append_extended_arch_args(extra_args: list[str], arch: Any) -> None:
    if arch.num_experts:
        extra_args.append(f"--num-experts {arch.num_experts}")
    if arch.moe_ffn_hidden_size:
        extra_args.append(f"--moe-ffn-hidden-size {arch.moe_ffn_hidden_size}")
    if arch.moe_shared_expert_intermediate_size:
        extra_args.append(
            f"--moe-shared-expert-intermediate-size {arch.moe_shared_expert_intermediate_size}"
        )
    if arch.moe_grouped_gemm:
        extra_args.append("--moe-grouped-gemm")
    if arch.moe_shared_expert_gate:
        extra_args.append("--moe-shared-expert-gate")
    if arch.moe_router_topk:
        extra_args.append(f"--moe-router-topk {arch.moe_router_topk}")
    if arch.moe_router_score_function:
        extra_args.append(
            f"--moe-router-score-function {arch.moe_router_score_function}"
        )
    if arch.moe_token_drop_policy:
        extra_args.append(f"--moe-token-drop-policy {arch.moe_token_drop_policy}")
    if arch.moe_router_dtype:
        extra_args.append(f"--moe-router-dtype {arch.moe_router_dtype}")
    if arch.moe_permute_fusion:
        extra_args.append("--moe-permute-fusion")
    if arch.moe_aux_loss_coeff is not None:
        extra_args.append(f"--moe-aux-loss-coeff {arch.moe_aux_loss_coeff}")
    if arch.megatron_spec:
        extra_args.append(f"--spec {' '.join(arch.megatron_spec)}")
    if arch.apply_layernorm_1p:
        extra_args.append("--apply-layernorm-1p")
    if arch.use_gated_attention:
        extra_args.append("--use-gated-attention")
    if arch.attention_output_gate:
        extra_args.append("--attention-output-gate")


def get_checkpoint_conversion_policy(
    cfg: Any,
    *,
    model: Any = None,
    single_rank_mtp: bool = False,
    extended_arch_args: bool = False,
    arch_args_model_script_attr: str | None = None,
) -> tuple[int, int, list[str]]:
    """Return (num_nodes, nproc_per_node, extra_args) for checkpoint conversion.

    ``single_rank_mtp`` forces a single-rank (tp=pp=world=1) layout for MTP/EAGLE
    checkpoints so the duplicated MTP head embedding/output isn't sharded (which
    corrupts the saved sharded state dict); training reshards on load.
    ``extended_arch_args`` emits the full MoE/attention arch flag set; when
    ``arch_args_model_script_attr`` is set, arch flags are skipped if that
    attribute is populated (the model script already sources them).
    """
    gpus_per_node = getattr(cfg, "conversion_gpus_per_node", None)
    if gpus_per_node is None:
        gpus_per_node = getattr(cfg, "actor_num_gpus_per_node", 8)
    if type(gpus_per_node) is not int or gpus_per_node <= 0:
        raise ValueError("conversion_gpus_per_node must be a positive integer")
    actor_nodes = getattr(cfg, "actor_num_nodes", 1)
    # torch_dist is reshard-friendly, so conversion parallelism is independent of
    # the training layout.
    tp = getattr(cfg, "conversion_tensor_model_parallel_size", None) or getattr(
        cfg, "tensor_model_parallel_size", 1
    )
    pp = getattr(cfg, "conversion_pipeline_model_parallel_size", None) or getattr(
        cfg, "pipeline_model_parallel_size", 1
    )

    if single_rank_mtp and tp == 1 and pp == 1 and getattr(cfg, "mtp_num_layers", 0):
        world_size = 1
    else:
        world_size = tp * pp if (tp > 1 or pp > 1) else gpus_per_node
    max_world_size = actor_nodes * gpus_per_node
    if world_size > max_world_size:
        raise ValueError(
            f"checkpoint conversion world_size={world_size} exceeds conversion cluster capacity "
            f"{actor_nodes}x{gpus_per_node}={max_world_size}"
        )

    for num_nodes in range(1, actor_nodes + 1):
        if world_size % num_nodes != 0:
            continue
        nproc_per_node = world_size // num_nodes
        if nproc_per_node > gpus_per_node:
            continue

        extra_args: list[str] = []
        if tp > 1 or pp > 1:
            extra_args += [
                f"--tensor-model-parallel-size {tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            if x := getattr(cfg, attr, None):
                extra_args.append(f"--{flag} {x}")

        emit_arch = bool(model and getattr(model, "architecture", None))
        if emit_arch and arch_args_model_script_attr:
            emit_arch = not getattr(cfg, arch_args_model_script_attr, "")
        if emit_arch:
            arch = model.architecture
            _append_common_arch_args(extra_args, arch)
            if extended_arch_args:
                _append_extended_arch_args(extra_args, arch)
            if arch.use_rotary_position_embeddings:
                extra_args.append("--use-rotary-position-embeddings")
                extra_args.append("--position-embedding-type rope")
                if extended_arch_args and arch.rotary_percent != 1.0:
                    extra_args.append(f"--rotary-percent {arch.rotary_percent}")

        return num_nodes, nproc_per_node, extra_args

    raise ValueError(
        f"cannot find checkpoint conversion layout for world_size={world_size} "
        f"with actor_num_nodes={actor_nodes}, actor_num_gpus_per_node={gpus_per_node}"
    )


def prepare_launch_config(
    cfg: Any,
    model: Any,
    tmpdir: str,
    *,
    yaml_config_fields: Any,
) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    import yaml

    if (
        model
        and not model.model_path
        and model.model_name
        and not str(model.model_name).startswith("/")
    ):
        model.model_path = resolve_checkpoint_ref(model.model_name)

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if val := getattr(cfg, attr, None):
            object.__setattr__(cfg, attr, resolve_checkpoint_ref(val))

    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            object.__setattr__(cfg, field, path)


def build_train_cmd(
    cfg: Any,
    root: str,
    *,
    model: Any = None,
    dataset: Any = None,
    model_script_attr: str,
) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    train_script = f"{root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    args = shlex.join(cfg.cli_args(dataset=dataset, model=model))
    if model_script := getattr(cfg, model_script_attr, ""):
        inner = (
            f"source {root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {args}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {args}"
