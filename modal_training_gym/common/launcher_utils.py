"""Framework-agnostic Modal launcher utilities shared by slime and miles.

Both frameworks' ``modal_helpers/utils.py`` are thin wrappers over these
implementations; they pass framework-specific parametrization (extended MoE
conversion args, the model-script attribute name, single-rank MTP handling) so
the shared logic stays in one place.
"""

from __future__ import annotations

import os
import shlex
from enum import Enum
from math import lcm
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any


def timing_debug_env() -> dict[str, str]:
    """Forward opt-in timing diagnostics to framework worker processes."""
    if os.environ.get("TRAINING_GYM_TIMING_DEBUG") == "1":
        return {"TRAINING_GYM_TIMING_DEBUG": "1"}
    return {}


# (attr_name_on_cfg, cli_flag) — optional per-rank conversion args
_CONVERSION_EXTRA_ARGS = [
    ("decoder_first_pipeline_num_layers", "decoder-first-pipeline-num-layers"),
    ("decoder_last_pipeline_num_layers", "decoder-last-pipeline-num-layers"),
    ("mtp_num_layers", "mtp-num-layers"),
    ("make_vocab_size_divisible_by", "make-vocab-size-divisible-by"),
    # The DeepSeek-V4 family selects its attention implementation here too: the
    # converter builds the model, and the default (megatron) rejects TP > 1.
    ("dsv4_impl", "dsv4-impl"),
]

_PIPELINE_SPLIT_ARGS = {
    "decoder_first_pipeline_num_layers",
    "decoder_last_pipeline_num_layers",
}

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
    for attr, flag in (
        ("kv_lora_rank", "kv-lora-rank"),
        ("qk_head_dim", "qk-head-dim"),
        ("qk_pos_emb_head_dim", "qk-pos-emb-head-dim"),
        ("v_head_dim", "v-head-dim"),
        ("rotary_scaling_factor", "rotary-scaling-factor"),
        ("mscale", "mscale"),
        ("mscale_all_dim", "mscale-all-dim"),
    ):
        if (value := getattr(arch, attr, None)) is not None and value != 0:
            extra_args.append(f"--{flag} {value}")
    for attr, flag in (
        ("no_masked_softmax_fusion", "no-masked-softmax-fusion"),
        ("multi_latent_attention", "multi-latent-attention"),
        ("no_rope_fusion", "no-rope-fusion"),
    ):
        if getattr(arch, attr, False):
            extra_args.append(f"--{flag}")
    if arch.num_experts:
        extra_args.append(f"--num-experts {arch.num_experts}")
    if arch.moe_layer_freq:
        extra_args.append(f"--moe-layer-freq {shlex.quote(str(arch.moe_layer_freq))}")
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
    if arch.moe_router_pre_softmax:
        extra_args.append("--moe-router-pre-softmax")
    if arch.moe_router_score_function:
        extra_args.append(
            f"--moe-router-score-function {arch.moe_router_score_function}"
        )
    if arch.moe_router_enable_expert_bias:
        extra_args.append("--moe-router-enable-expert-bias")
    if arch.moe_router_load_balancing_type:
        extra_args.append(
            f"--moe-router-load-balancing-type {arch.moe_router_load_balancing_type}"
        )
    if arch.moe_token_dispatcher_type:
        extra_args.append(
            f"--moe-token-dispatcher-type {arch.moe_token_dispatcher_type}"
        )
    if arch.moe_router_bias_update_rate is not None:
        extra_args.append(
            f"--moe-router-bias-update-rate {arch.moe_router_bias_update_rate}"
        )
    if arch.moe_router_group_topk:
        extra_args.append(f"--moe-router-group-topk {arch.moe_router_group_topk}")
    if arch.moe_router_num_groups:
        extra_args.append(f"--moe-router-num-groups {arch.moe_router_num_groups}")
    if arch.moe_router_topk_scaling_factor is not None:
        extra_args.append(
            f"--moe-router-topk-scaling-factor {arch.moe_router_topk_scaling_factor}"
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

    TP/PP come from the ``conversion_*`` overrides when set, else from the training
    layout. Explicit TP1/PP1 uses one rank instead of the automatic node-wide
    conversion. EP/ETP are emitted only when their ``conversion_*`` fields are set
    explicitly, and the pipeline-split args are dropped at conversion PP1. An EP
    wider than tp*pp widens the world with data-parallel replicas.
    """
    gpus_per_node = getattr(cfg, "actor_num_gpus_per_node", 8)
    actor_nodes = getattr(cfg, "actor_num_nodes", 1)
    # torch_dist is reshard-friendly, so conversion parallelism is independent of
    # the training layout.
    tp = getattr(cfg, "conversion_tensor_model_parallel_size", None) or getattr(
        cfg, "tensor_model_parallel_size", 1
    )
    pp = getattr(cfg, "conversion_pipeline_model_parallel_size", None) or getattr(
        cfg, "pipeline_model_parallel_size", 1
    )
    pins_layout = (
        tp > 1
        or pp > 1
        or getattr(cfg, "conversion_tensor_model_parallel_size", None) is not None
        or getattr(cfg, "conversion_pipeline_model_parallel_size", None) is not None
    )
    # Expert parallelism is opt-in: torch_dist reshards, so most models convert fine
    # at the implicit EP1. Models whose full expert set does not fit a rank at EP1
    # (Inkling-Small: 256 experts) set it explicitly to mirror upstream's layout.
    ep = getattr(cfg, "conversion_expert_model_parallel_size", None)
    etp = getattr(cfg, "conversion_expert_tensor_parallel_size", None)
    if (ep or etp) and not (ep and etp):
        raise ValueError(
            "checkpoint conversion expert parallelism needs both "
            "conversion_expert_model_parallel_size and "
            f"conversion_expert_tensor_parallel_size (got ep={ep}, etp={etp}); "
            "Megatron otherwise defaults ETP to TP and the expert world size no "
            "longer matches tp*pp"
        )
    if ep and etp and not pins_layout:
        raise ValueError(
            "checkpoint conversion expert parallelism needs a pinned conversion "
            "layout: set conversion_tensor_model_parallel_size or "
            "conversion_pipeline_model_parallel_size, otherwise the converter "
            "inflates PP to the rank count and the expert world size no longer "
            "divides it"
        )
    if single_rank_mtp and tp == 1 and pp == 1 and getattr(cfg, "mtp_num_layers", 0):
        world_size = 1
    else:
        world_size = tp * pp if pins_layout else gpus_per_node

    if ep and etp:
        # The expert world can be wider than tp*pp: Megatron forms expert groups
        # over the (dp x tp) ranks, so a larger EP is served by data-parallel
        # replicas rather than by more tensor or pipeline shards. Widening the
        # world is what makes a big MoE fit — every replica holds only its
        # 1/EP slice of the experts.
        world_size = lcm(world_size, etp * ep * pp)
    max_world_size = actor_nodes * gpus_per_node
    if world_size > max_world_size:
        raise ValueError(
            f"checkpoint conversion world_size={world_size} exceeds actor cluster capacity "
            f"{actor_nodes}x{gpus_per_node}={max_world_size}"
        )

    for num_nodes in range(1, actor_nodes + 1):
        if world_size % num_nodes != 0:
            continue
        nproc_per_node = world_size // num_nodes
        if nproc_per_node > gpus_per_node:
            continue

        extra_args: list[str] = []
        if pins_layout:
            extra_args += [
                f"--tensor-model-parallel-size {tp}",
                f"--pipeline-model-parallel-size {pp}",
            ]
        if ep:
            extra_args.append(f"--expert-model-parallel-size {ep}")
        if etp:
            extra_args.append(f"--expert-tensor-parallel-size {etp}")
        for attr, flag in _CONVERSION_EXTRA_ARGS:
            # A recipe carries one ``decoder_{first,last}_pipeline_num_layers`` pair
            # shared by conversion and training. It describes a split across >1
            # pipeline stages, so it is meaningless (and rejected by Megatron) when
            # the conversion layout is PP1. Only drop it when that PP1 is pinned on the
            # command line: with no layout flags the converter picks its own PP, and
            # dropping the split would then describe a layout nobody asked for.
            if pp == 1 and pins_layout and attr in _PIPELINE_SPLIT_ARGS:
                continue
            # Zero is meaningful for MTP: it overrides model-script defaults
            # during conversion just as the recipe does during training.
            if (x := getattr(cfg, attr, None)) is not None:
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


def serialize_recipe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path | PurePosixPath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize_recipe_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [serialize_recipe_value(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module}.{name}" if module and name else repr(value)
    return repr(value)


REDACTED = "[redacted]"

# Substrings that mark a name as credential-bearing. Matched against the
# lowercased name, so upper-case env var names in ``train_env_vars``
# (``HF_TOKEN``, ``AWS_SECRET_ACCESS_KEY``) and header keys in
# ``sglang_request_params`` (``Authorization``) are covered too.
#
# Deliberately substring rather than exact: the recipe dataclasses declare no
# credential fields at all today (the only key/token-ish names are
# ``multimodal_keys``, ``max_tokens_per_gpu``, ``calculate_per_token_loss`` and
# ``rollout_stop_token_ids`` — none secret). Every real credential arrives
# through a free-form dict (``extra_config``, ``train_env_vars``,
# ``sglang_config``, ``eval_config``, ``sglang_request_params``) under a key
# name we do not control, so recall matters more than brevity here.
_SENSITIVE_NAME_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "wandb_key",
    "secret",
    "password",
    "passwd",
    "credential",
    "token",
    "bearer",
    "authorization",
    "oauth",
)

# Names containing "token" that are counts, limits, policies or vocabulary ids
# rather than credentials. Without these, ``max_tokens_per_gpu`` and friends get
# redacted and the dashboard parameter table loses real tuning knobs.
_TOKEN_NOT_A_CREDENTIAL = (
    "token_id",
    "per_token",
    "token_per",
    "tokens_per",
    "num_token",
    "max_token",
    "min_token",
    "token_count",
    "token_len",
    "token_limit",
    "token_budget",
    "token_ratio",
    "token_loss",
    "tokenizer",
)

# Value prefixes distinctive enough to identify a credential regardless of the
# key it is filed under — the backstop for a secret in a free-form dict whose
# key name says nothing (e.g. ``{"headers": {"x-upstream": "sk-…"}}``).
_SENSITIVE_VALUE_PREFIXES = (
    "wk-",  # Modal proxy-auth key id
    "ws-",  # Modal proxy-auth key secret
    "hf_",  # Hugging Face user access token
    "sk-",  # OpenAI-style API key
    "ghp_",  # GitHub personal access token
    "gho_",
    "github_pat_",
    "xox",  # Slack bot/user/app token
    "akia",  # AWS access key id
    "asia",
)


def is_sensitive_recipe_field(name: str) -> bool:
    normalized = name.lower()
    if "token" in normalized and any(
        benign in normalized for benign in _TOKEN_NOT_A_CREDENTIAL
    ):
        return False
    return any(marker in normalized for marker in _SENSITIVE_NAME_MARKERS)


def is_sensitive_recipe_value(value: Any) -> bool:
    """Whether a value looks like a credential on its own, ignoring its key.

    Requires a recognizable prefix *and* opaque-secret shape (long, no spaces or
    path separators) so ordinary settings that happen to share a prefix — an
    ``hf_``-prefixed flag name, a ``/root/...`` path — are not redacted.
    """
    if not isinstance(value, str) or len(value) < 20:
        return False
    if any(ch in value for ch in " \t/\\"):
        return False
    lowered = value.lower()
    return any(lowered.startswith(p) for p in _SENSITIVE_VALUE_PREFIXES)


def serialize_recipe_param_value(name: str, value: Any) -> Any:
    if is_sensitive_recipe_field(name):
        return REDACTED if value not in (None, "", False) else value
    if isinstance(value, dict):
        return {
            str(k): serialize_recipe_param_value(str(k), v) for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [serialize_recipe_param_value(name, v) for v in value]
    if is_sensitive_recipe_value(value):
        return REDACTED
    return serialize_recipe_value(value)


def serialize_recipe_params(
    recipe: Any,
    *,
    dataset: Any = None,
    eval_dataset: Any = None,
    dataset_path: str | None = None,
    eval_dataset_path: str | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """The recipe's effective CLI flags, serialized for the dashboard run record.

    Framework-agnostic: any recipe implementing ``_fields`` works, so slime and
    miles runs get the same parameter table.
    """
    return {
        key: serialize_recipe_param_value(key, value)
        for key, value in recipe._fields(
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            model=model,
        ).items()
    }


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

    escape_hatch = getattr(cfg, "_ESCAPE_HATCH_FIELD", None)
    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            print(f"Materialized {field} → {path}")
            # Record the keys before the dict is replaced by its path: this runs
            # before build_train_cmd calls cli_args, and the recipe's
            # escape-hatch-wins-over-flag rule needs to know what was in there
            # (see BaseTrainRecipe._escape_hatch_keys).
            if field == escape_hatch:
                object.__setattr__(cfg, "_materialized_config_keys", tuple(val))
                object.__setattr__(cfg, "_materialized_config", dict(val))
            object.__setattr__(cfg, field, path)


def drop_materialized_config_key(cfg: Any, key: str) -> None:
    """Remove ``key`` from the recipe's already-materialized escape-hatch YAML.

    Runs after ``prepare_launch_config`` has replaced the escape-hatch dict
    with a file path, so the YAML is rewritten and the recorded keys updated
    so ``_emit_fields`` stops suppressing the same-named flag.
    """
    import yaml

    escape_hatch = getattr(cfg, "_ESCAPE_HATCH_FIELD", None)
    if not escape_hatch:
        return
    keys = tuple(getattr(cfg, "_materialized_config_keys", ()) or ())
    if key not in keys:
        return
    path = getattr(cfg, escape_hatch, None)
    if isinstance(path, str) and os.path.isfile(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        data.pop(key, None)
        with open(path, "w") as f:
            yaml.dump(data, f)
    stored = getattr(cfg, "_materialized_config", None)
    if isinstance(stored, dict):
        stored = dict(stored)
        stored.pop(key, None)
        object.__setattr__(cfg, "_materialized_config", stored)
    object.__setattr__(
        cfg, "_materialized_config_keys", tuple(k for k in keys if k != key)
    )


def build_train_cmd(
    cfg: Any,
    root: str,
    *,
    model: Any = None,
    dataset: Any = None,
    eval_dataset: Any = None,
    dataset_path: str | None = None,
    eval_dataset_path: str | None = None,
    model_script_attr: str,
    model_args_command: str = "",
) -> str:
    """Build the Ray job entrypoint, sourcing model arch args if needed."""
    train_script = f"{root}/{'train_async.py' if cfg.train_async else 'train.py'}"
    args = shlex.join(
        cfg.cli_args(
            dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_path=dataset_path,
            eval_dataset_path=eval_dataset_path,
            model=model,
        )
    )
    if model_script := getattr(cfg, model_script_attr, ""):
        inner = (
            f"source {root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {args}"
        )
        return f"bash -c {shlex.quote(inner)}"
    if model_args_command:
        inner = (
            f'MODEL_ARGS_LINE="$({model_args_command})" '
            f'|| exit 1; read -ra MODEL_ARGS <<< "$MODEL_ARGS_LINE"; '
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {args}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {args}"
