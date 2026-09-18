"""Factory that builds a Modal app for a slime training run from config objects.

Usage (from a tutorial file):

    from modal_training_gym.common.train import TrainConfig
    from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

    config = TrainConfig(
        model=my_model,
        dataset=my_dataset,
        recipe=SlimeRecipe(...),
    )
    app = config.build_app()

Then: `uv run modal run <tutorial_file>.py::train`.
"""

import asyncio
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Callable, Mapping
from enum import Enum
from modal import (
    App,
    Dict as ModalDict,
    Image,
    Retries,
    Secret,
    Volume,
    current_function_call_id,
)

from modal_training_gym.common import hf_secrets


from modal_training_gym.common.dataset import DatasetConfig, HarborDataset
from modal_training_gym.common.framework import (
    mount_tools_dir,
)
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.ray_cluster import (
    ModalRayCluster,
    _supports_rdma,
    capture_ray_cluster_diagnostics,
    clustered_if,
)
from modal_training_gym.common.run import (
    TrainingRunStatus,
    has_torch_dist_checkpoint,
    mark_training_attempt_finished,
    record_training_attempt_cluster_identity,
    record_wandb_attempt,
    record_resume_checkpoint,
    select_accepted_wandb_attempt,
    torch_dist_resume_checkpoint,
)
from modal_training_gym.common.attempts import (
    RUN_CONTRACT_SCHEMA_VERSION,
    attempt_root,
    create_attempt_namespace,
    load_latest_committed_boundary,
    run_contract_sha256,
    write_accepted_lineage,
)
from modal_training_gym.common.launcher_helpers import (
    AcceptedTrainResultError,
    bind_accepted_train_result,
    build_app_tags,
    build_terminal_run_record,
    build_train_result,
    capture_and_record_ray_failure_diagnostic,
    compute_save_root,
    init_training_run_record,
    load_accepted_train_result,
    mark_run_failed,
    mark_run_stopped,
    record_attempt_failure,
    record_setup_failure,
    record_last_committed_boundary_snapshot,
    record_ray_failure_diagnostic,
    resolve_caller_context,
    resolve_checkpoint_volumes,
    run_download_phase,
    run_prepare_dataset,
    ship_callable,
)
from modal_training_gym.common.wandb import (
    WandbConfig,
    build_wandb_runtime_env,
    install_wandb_api_key_in_process,
)
from modal_training_gym.common.launcher_utils import redact_runtime_env
from modal_training_gym.common.status import SlimeStatus

from modal_training_gym.train_recipes.slime_recipe.recipe import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeRecipe,
)
from .modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
    get_modal_cluster_context,
    prepare_slime_config,
    resolve_checkpoint_ref,
)
from modal_training_gym.common.patches import encode_patch
from modal_training_gym.common.checkpoint import Checkpoint
from modal_training_gym.common.framework import Framework


def _capture_remote_entry_clock(environment: Mapping[str, str]) -> dict[str, str]:
    """Capture the Modal entry clock before cluster or attempt initialization."""

    receipt_sha256 = environment.get("DRIFT_ASYNC_RL_OPERATOR_POOL_RECEIPT_SHA256", "")
    if not receipt_sha256:
        return {}
    if len(receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in receipt_sha256
    ):
        raise ValueError("operator-pool receipt hash is malformed at remote entry")
    return {
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_EPOCH_NS": str(time.time_ns()),
        "DRIFT_ASYNC_RL_REMOTE_ENTRY_RECEIPT_SHA256": receipt_sha256,
    }


def _remote_entry_runtime_env(
    environment: Mapping[str, str],
    remote_entry_clock: Mapping[str, str],
) -> dict[str, str]:
    """Propagate the sealed entry clock without allowing recipe overwrite."""

    return {**environment, **remote_entry_clock}


def _remote_execution_identity(modal_app_id: str) -> dict[str, str]:
    """Bind Ray workers to the actual Modal app/function invocation."""

    function_call_id = str(current_function_call_id() or "")
    if not modal_app_id or not function_call_id:
        raise RuntimeError("remote Modal app/function-call identity is unavailable")
    return {
        "TRAINING_GYM_MODAL_APP_ID": modal_app_id,
        "TRAINING_GYM_FUNCTION_CALL_ID": function_call_id,
    }


SLIME_ROOT = "/root/slime"
# Pin by digest to prevent mutable-tag drift.  Tag: nightly-dev-20260703b
SLIME_IMAGE = "slimerl/slime@sha256:269b44b17e3f7136447db4cdaa3bf36ef9e3169f1596af0d7180c45f2a301965"
# v0.8.0+ makes per-task CPU/memory requests configurable via enforcement
# policies ("limit"/"ignore"), letting sandboxes burst on Modal and bill by
# actual CPU-/RAM-second usage instead of over-provisioning a static reservation.
HARBOR_PKG_VERSION = "0.8.0"
# Pin the complete small dependency closure; do not introduce a mutable
# installer image into scientific runtime builds.
READABLE_ID_PACKAGES = (
    "randomname==0.2.1",
    "fire==0.7.1",
    "termcolor==3.3.0",
)
READABLE_ID_INSTALL_COMMAND = (
    f"python3 -m pip install {shlex.join(READABLE_ID_PACKAGES)}"
)


def _modal_retry_policy(max_retries: int) -> Retries | None:
    """Represent zero retries with Modal's unambiguous no-policy default."""

    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError("max_retries must be an integer")
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    return (
        None
        if max_retries == 0
        else Retries(max_retries=max_retries, initial_delay=0.0)
    )


def _validate_attempt_limit(attempt_count: int, max_attempts: int | None) -> None:
    """Fail before Ray/model work when a logical-attempt ceiling was exceeded."""

    if max_attempts is not None and attempt_count > max_attempts:
        raise RuntimeError(
            "logical attempt limit exceeded before Ray bootstrap: "
            f"attempt_count={attempt_count} max_attempts={max_attempts}"
        )


@dataclasses.dataclass(frozen=True)
class _AcceptedTrainResultReplay:
    """Authenticated result returned without creating another attempt."""

    payload: dict[str, Any] | None


_SLIME_PATCHES = Path(__file__).parent / "modal_helpers" / "patches"
_PATCH_VALIDATION_B64 = encode_patch("patch_validation", _SLIME_PATCHES)
_PATCH_MEGATRON_BRIDGE_B64 = encode_patch("patch_megatron_bridge", _SLIME_PATCHES)
_PATCH_TORCH_LOAD_B64 = encode_patch("patch_torch_load", _SLIME_PATCHES)
_PATCH_GLOBAL_PLAN_B64 = encode_patch("patch_global_plan", _SLIME_PATCHES)
_PATCH_CHECKPOINT_SAVE_B64 = encode_patch("patch_checkpoint_save", _SLIME_PATCHES)
_PATCH_ADVANTAGES_B64 = encode_patch("patch_advantages", _SLIME_PATCHES)
_PATCH_BRIDGE_NONE_TASK_B64 = encode_patch("patch_bridge_none_task", _SLIME_PATCHES)
_PATCH_GDN_PACKED_SEQ_B64 = encode_patch("patch_gdn_packed_seq", _SLIME_PATCHES)
_PATCH_BRIDGE_PER_TOKEN_LOSS_B64 = encode_patch(
    "patch_bridge_provider_per_token_loss", _SLIME_PATCHES
)
# The Qwen3-ASR Megatron->HF converter (registers the qwen3_asr mapping incl. the
# audio tower). It lives in the base image — not the ASR recipe — because torch_dist
# -> HF conversion runs in the shared convert_checkpoint_to_hf path (deploy/eval),
# which has no recipe; baking it here makes both train-time export and deploy-time
# conversion ASR-capable. Additive + idempotent, so non-ASR runs are untouched.
_PATCH_QWEN3_ASR_EXPORT_B64 = encode_patch(
    "patch_qwen3_asr_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_asr",
)
# The Qwen3-VL Megatron->HF converters: a qwen3_vl per-param mapping (language
# stack + frozen-ViT identity passthrough) and a torch_dist->HF shim that skips
# the frozen ViT's stacked layers.
_PATCH_QWEN3_VL_EXPORT_B64 = encode_patch(
    "patch_qwen3_vl_export",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_QWEN3_VL_TORCH_DIST_B64 = encode_patch(
    "patch_qwen3_vl_torch_dist",
    _SLIME_PATCHES / "model_specific_patches" / "qwen3_vl",
)
_PATCH_ROLLOUT_STATUS_B64 = encode_patch(
    "patch_rollout_status_reporting", _SLIME_PATCHES
)
_PATCH_ADVANTAGE_DIST_B64 = encode_patch("patch_advantage_distribution", _SLIME_PATCHES)
_PATCH_LOG_ELIDE_B64 = encode_patch("patch_log_elide", _SLIME_PATCHES)
# Backport of NVIDIA/Megatron-LM #3845: dequantize quantized CUDA tensors in the
# async dist-checkpoint writer before serialization. slime pins a pre-#3845
# Megatron, so FP8/TE _extra_state tensors otherwise crash the torch_dist save
# with inline_container.cc "unexpected pos" (e.g. the GLM-5.2 convert). No-op for
# non-quantized tensors, so safe for every image.
_PATCH_DIST_CKPT_QUANTIZED_B64 = encode_patch(
    "patch_dist_ckpt_quantized", _SLIME_PATCHES
)
# USACO VPO patch: threads the per-test reward vector into rollout_data for the VPO custom advantage fn.
_PATCH_VPO_ROLLOUT_DATA_B64 = encode_patch("patch_vpo_rollout_data", _SLIME_PATCHES)
_PATCH_MEGAGEM_ROLLOUT_DATA_B64 = encode_patch(
    "patch_megagem_rollout_data", _SLIME_PATCHES
)


def _build_slime_base_image() -> "Image":
    return (
        Image.from_registry(SLIME_IMAGE)
        .entrypoint([])
        .run_commands(
            "rm -rf /root/.cache/huggingface",
            f"echo {_PATCH_MEGATRON_BRIDGE_B64} | base64 -d | python3",
            f"echo {_PATCH_ADVANTAGES_B64} | base64 -d | python3",
            f"echo {_PATCH_BRIDGE_NONE_TASK_B64} | base64 -d | python3",
            f"echo {_PATCH_QWEN3_ASR_EXPORT_B64} | base64 -d | python3",
            f"echo {_PATCH_QWEN3_VL_EXPORT_B64} | base64 -d | python3",
            f"echo {_PATCH_QWEN3_VL_TORCH_DIST_B64} | base64 -d | python3",
            f"echo {_PATCH_ROLLOUT_STATUS_B64} | base64 -d | python3",
            f"echo {_PATCH_ADVANTAGE_DIST_B64} | base64 -d | python3",
            f"echo {_PATCH_LOG_ELIDE_B64} | base64 -d | python3",
            f"echo {_PATCH_DIST_CKPT_QUANTIZED_B64} | base64 -d | python3",
            f"echo {_PATCH_VPO_ROLLOUT_DATA_B64} | base64 -d | python3",
            f"echo {_PATCH_MEGAGEM_ROLLOUT_DATA_B64} | base64 -d | python3",
        )
    )


def _add_training_gym_runtime(image: "Image") -> "Image":
    """Ship runtime code and pinned ID dependencies before recipe environment."""

    return image.add_local_python_source("modal_training_gym", copy=True).run_commands(
        READABLE_ID_INSTALL_COMMAND
    )


def _build_conversion_config(slime_cfg: Any, model: Any = None) -> dict[str, Any]:
    """Build a dict of parameters that affect the torch_dist checkpoint layout.

    Written to ``.conversion_config.json`` inside the checkpoint directory after
    conversion so that later runs can detect stale checkpoints whose parallelism
    no longer matches the current recipe.
    """
    from modal_training_gym.frameworks.slime.modal_helpers.utils import (
        get_checkpoint_conversion_policy,
    )

    num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
        slime_cfg, model=model
    )
    return {
        "num_nodes": num_nodes,
        "nproc_per_node": nproc_per_node,
        "extra_args": extra_args,
        "model_name": model.model_name if model else None,
        "slime_model_script": getattr(slime_cfg, "slime_model_script", ""),
    }


_CONVERSION_CONFIG_FILE = ".conversion_config.json"


def _response_parser_path(model: Any) -> str:
    """Import path of the model's response parser so the rollout recorder can
    resolve and apply it remotely. Empty when the model sets no parser."""
    fn = getattr(model, "response_parser", None) if model is not None else None
    if fn is None:
        return ""
    module = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "")
    return f"{module}.{qualname}" if module and qualname else ""


def _is_complete_torch_dist_checkpoint(path: str) -> bool:
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return "common.pt" in names and any(name.endswith(".distcp") for name in names)


def _serialize_recipe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path | PurePosixPath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_recipe_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_serialize_recipe_value(v) for v in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return f"{module}.{name}" if module and name else repr(value)
    return repr(value)


def _is_sensitive_recipe_field(name: str) -> bool:
    normalized = name.lower()
    if normalized.endswith("token_id") or normalized.endswith("token_ids"):
        return False
    return (
        any(
            marker in normalized
            for marker in ("api_key", "access_key", "secret", "password")
        )
        or normalized in {"token", "auth", "authorization", "cookie", "wandb_key"}
        or normalized.endswith(("_token", "_auth", "_authorization", "_cookie"))
        or normalized == "wandb_key"
    )


def _serialize_slime_param_value(name: str, value: Any) -> Any:
    if _is_sensitive_recipe_field(name):
        return "[redacted]" if value not in (None, "", False) else value
    if isinstance(value, dict):
        return {
            str(k): _serialize_slime_param_value(str(k), v) for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_serialize_slime_param_value(name, v) for v in value]
    if isinstance(value, set | frozenset):
        converted = [_serialize_slime_param_value(name, v) for v in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    return _serialize_recipe_value(value)


def _serialize_slime_params(
    recipe: SlimeRecipe,
    *,
    dataset: DatasetConfig | None = None,
    model: ModelConfig | None = None,
) -> dict[str, Any]:
    serialized = {
        key: _serialize_slime_param_value(key, value)
        for key, value in recipe._fields(dataset=dataset, model=model).items()
    }
    # ``attempt_mode`` is intentionally not a Slime CLI flag, but it is part
    # of the durable reporting contract. In particular the first attempt-start
    # event is written before metadata["attempt_mode"] is populated later in
    # launcher setup, so the config snapshot must carry this discriminator.
    serialized["attempt_mode"] = recipe.attempt_mode
    return serialized


def _setup_rank_owns_logical_run_failure(is_head: bool | None) -> bool:
    """Only a positively discovered rank 0 may terminalize the logical run."""
    return is_head is True


def _contract_value(value: Any, *, field_name: str = "") -> Any:
    """Convert a scientific config value to deterministic, credential-safe JSON."""
    if field_name and _is_sensitive_recipe_field(field_name):
        return "[redacted]" if value not in (None, "", False) else value
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        # ``run_contract_sha256`` rejects NaN/Infinity via allow_nan=False.
        return value
    if isinstance(value, Enum):
        return _contract_value(value.value, field_name=field_name)
    if isinstance(value, Path | PurePosixPath):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _contract_value(dataclasses.asdict(value), field_name=field_name)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _contract_value(model_dump(mode="json"), field_name=field_name)
    if isinstance(value, Mapping):
        return {
            str(key): _contract_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_contract_value(item, field_name=field_name) for item in value]
    if isinstance(value, set | frozenset):
        converted = [_contract_value(item, field_name=field_name) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        if module and name:
            return {"callable": f"{module}.{name}"}
    raise TypeError(
        "scientific run contract contains a non-canonical value "
        f"for {field_name or '<root>'}: {type(value).__name__}"
    )


def _public_config_snapshot(value: Any) -> dict[str, Any]:
    """Capture public declarative class defaults plus instance overrides."""
    names: set[str] = set()
    for cls in reversed(type(value).__mro__):
        for name, raw in vars(cls).items():
            if (
                name.startswith("_")
                or isinstance(raw, property | staticmethod | classmethod)
                or callable(raw)
            ):
                continue
            names.add(name)
    names.update(name for name in vars(value) if not name.startswith("_"))
    snapshot: dict[str, Any] = {}
    for name in sorted(names):
        raw = getattr(value, name)
        if callable(raw):
            module = getattr(raw, "__module__", "")
            qualname = getattr(raw, "__qualname__", getattr(raw, "__name__", ""))
            snapshot[name] = {"callable": f"{module}.{qualname}"}
        else:
            snapshot[name] = _contract_value(raw, field_name=name)
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "fields": snapshot,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash the exact regular files copied into a source-backed image layer."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"source tree does not exist: {root}")
    ignored = {".git", ".venv", "__pycache__"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not any(part in ignored for part in path.relative_to(root).parts)
        and path.suffix != ".pyc"
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _image_overlay_contract(
    overlay: Any,
    source_roots: list[str],
    *,
    required: bool,
) -> dict[str, Any] | None:
    """Bind a custom image overlay and every local source tree it copies.

    Modal's image object does not expose a portable content digest while the
    app is being constructed. Committed mode therefore requires the caller to
    declare local overlay inputs explicitly. The contract stores hashes only,
    never local absolute paths or file contents.
    """
    if overlay is None:
        if source_roots:
            raise ValueError(
                "image_overlay_source_roots were provided without image_overlay"
            )
        return None
    if required and not source_roots:
        raise ValueError(
            "committed attempt mode requires image_overlay_source_roots so the "
            "custom image content is bound into the scientific run contract"
        )

    module = str(getattr(overlay, "__module__", "") or "")
    qualname = str(
        getattr(overlay, "__qualname__", getattr(overlay, "__name__", "")) or ""
    )
    source_file_receipt = None
    try:
        source_file = Path(inspect.getsourcefile(overlay) or "")
    except (TypeError, OSError):
        source_file = Path()
    if source_file.is_file():
        source_file_receipt = {
            "name": source_file.name,
            "sha256": _file_sha256(source_file),
        }
    if required and (not module or not qualname or source_file_receipt is None):
        raise ValueError(
            "committed image_overlay must be a source-backed callable with a "
            "stable module and qualified name"
        )

    roots: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for raw_root in source_roots:
        root = Path(raw_root)
        if not root.exists():
            raise ValueError(f"image overlay source root does not exist: {root}")
        label = root.name
        if not label or label in seen_labels:
            raise ValueError(
                "image_overlay_source_roots must have unique nonempty basenames"
            )
        seen_labels.add(label)
        if root.is_file() and not root.is_symlink():
            roots.append(
                {
                    "label": label,
                    "kind": "file",
                    "sha256": _file_sha256(root),
                }
            )
        elif root.is_dir() and not root.is_symlink():
            roots.append(
                {
                    "label": label,
                    "kind": "directory",
                    "sha256": _source_tree_sha256(root),
                }
            )
        else:
            raise ValueError(
                "image overlay source roots must be regular files or directories"
            )
    return {
        "callable": f"{module}.{qualname}" if module and qualname else "",
        "source_file": source_file_receipt,
        "source_roots": sorted(roots, key=lambda item: item["label"]),
    }


def _scientific_run_contract(
    *,
    training_run_id: str,
    recipe: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None,
    caller_script: str | None,
    has_hybrid_spec: bool,
    has_gdn: bool,
    train_ephemeral_disk: Any,
    train_timeout_seconds: int,
    train_experimental_options: Mapping[str, Any],
    image_overlay_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the immutable contract that every retry boundary must match."""
    patch_receipts = []
    for raw_path in recipe.patch_files:
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"scientific image patch does not exist: {path}")
        patch_receipts.append(
            {
                "remote_name": path.name,
                "sha256": _file_sha256(path),
            }
        )
    caller_receipt = None
    if caller_script is not None:
        caller_path = Path(caller_script)
        if not caller_path.is_file():
            raise ValueError(f"scientific caller source does not exist: {caller_path}")
        caller_receipt = {
            "remote_name": caller_path.name,
            "sha256": _file_sha256(caller_path),
        }
    local_slime_receipt = (
        {
            "source_tree_sha256": _source_tree_sha256(Path(recipe.local_slime)),
        }
        if recipe.local_slime
        else None
    )
    checkpoint_identity = (
        {
            "checkpoint_type": checkpoint.checkpoint_type.value,
            "name": checkpoint.name,
            "path": checkpoint.path,
            "training_run_id": checkpoint.training_run_id,
            "checkpoints_volume_name": checkpoint.checkpoints_volume_name,
            "checkpoints_mount_path": checkpoint.checkpoints_mount_path,
        }
        if checkpoint is not None
        else None
    )
    payload = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "framework": "slime",
        "training_run_id": training_run_id,
        "effective_train_fields": _contract_value(
            _serialize_slime_params(recipe, dataset=dataset, model=model)
        ),
        "retry_policy": {
            "attempt_mode": recipe.attempt_mode,
            "max_retries": int(recipe.max_retries),
            "max_attempts": recipe.max_attempts,
            "save_interval": int(recipe.save_interval),
        },
        "model": _public_config_snapshot(model),
        "dataset": _public_config_snapshot(dataset),
        "initial_checkpoint": checkpoint_identity,
        "image": {
            "base_image": SLIME_IMAGE,
            "training_gym_source_sha256": _source_tree_sha256(
                Path(__file__).resolve().parents[2]
            ),
            "image_environment": _contract_value(recipe.image_env),
            "runtime_environment": _contract_value(recipe.environment),
            "image_run_commands": list(recipe.image_run_commands),
            "image_overlay": image_overlay_contract,
            "patches": patch_receipts,
            "caller_source": caller_receipt,
            "local_slime": local_slime_receipt,
            "built_in_patch_profile": {
                "hybrid": bool(has_hybrid_spec),
                "gdn": bool(has_gdn),
                "bridge": recipe.megatron_to_hf_mode == "bridge",
            },
        },
        "modal_execution": {
            "gpu_type": recipe.gpu_type,
            "total_nodes": int(recipe.total_nodes),
            "actor_num_nodes": int(recipe.actor_num_nodes),
            "actor_num_gpus_per_node": int(recipe.actor_num_gpus_per_node),
            "physical_gpus_per_node": int(recipe.gpu_allocation.gpus_per_node),
            "rollout_num_gpus": int(recipe.rollout_num_gpus or 0),
            "memory": _contract_value(recipe.memory),
            "cloud": recipe.cloud,
            "region": recipe.region,
            "ephemeral_disk": _contract_value(train_ephemeral_disk),
            "timeout_seconds": train_timeout_seconds,
            "experimental_options": _contract_value(train_experimental_options),
        },
    }
    # Round-trip now so an unsupported object or non-finite numeric fails before
    # a claim-grade attempt can allocate GPUs or inspect an earlier boundary.
    run_contract_sha256(payload)
    return payload


def _preflight_wandb(wandb_cfg: WandbConfig) -> str:
    """Thin wrapper around :func:`~modal_training_gym.common.wandb.preflight_wandb`."""
    from modal_training_gym.common.wandb import preflight_wandb

    return preflight_wandb(wandb_cfg)


def _validate_committed_dataset_inputs(
    slime: SlimeRecipe,
    dataset: DatasetConfig,
) -> None:
    if slime.attempt_mode == "committed" and dataset.always_prepare:
        raise ValueError(
            "committed attempt mode requires dataset.always_prepare=False: "
            "deleting and rebuilding shared materialized data on retry would "
            "invalidate an authenticated parent boundary"
        )


def _pop_train_function_timeout(train_function_kwargs: dict[str, Any]) -> int:
    """Remove and validate the per-run Modal function timeout.

    Keeping this value recipe-controlled lets short, hard-capped scientific jobs
    fail closed well before the framework's extended-horizon default.  A bool is
    rejected explicitly because it is an ``int`` subclass but cannot be a
    meaningful duration here.
    """

    timeout = train_function_kwargs.pop("timeout", 48 * 60 * 60)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise TypeError(
            "slime.train_function_kwargs['timeout'] must be a positive integer"
        )
    return timeout


async def _require_d1a_function_call_binding(
    run_record: Any,
    *,
    polls: int = 15 * 60 * 4,
    poll_seconds: float = 0.25,
) -> Any:
    """Join the local spawn ID into the remote run object before paid work.

    ``TrainConfig.launch`` only learns Modal's FunctionCall ID after spawning
    this function.  The remote initializer can therefore load an earlier record
    with an empty ID and later overwrite the local binding from stale memory.
    Receipt-bound D1a runs wait for the local writer, then copy its nonempty ID
    into the long-lived remote object so every later status save preserves it.
    The default 15-minute join allowance matches the sealed local cold-start
    allowance; a queued H200 container must not fail provenance after 30 seconds.
    """

    if os.environ.get("DRIFT_ASYNC_RL_D1_MATRIX") != "1":
        return run_record
    if isinstance(polls, bool) or not isinstance(polls, int) or polls < 1:
        raise ValueError("D1a function-binding polls must be a positive integer")
    if poll_seconds < 0:
        raise ValueError("D1a function-binding poll interval must be nonnegative")
    record_type = type(run_record)
    last_transient_error: Exception | None = None
    for index in range(polls):
        try:
            latest = await record_type.from_id(
                run_record.training_run_id,
                is_async=True,
            )
        except Exception as exc:
            last_transient_error = exc
            latest = None
        if latest is not None:
            latest_run_id = str(getattr(latest, "training_run_id", "") or "")
            latest_app_id = str(getattr(latest, "modal_app_id", "") or "")
            if latest_run_id != str(run_record.training_run_id):
                raise RuntimeError(
                    "D1a persisted TrainingRun identity changed during remote startup"
                )
            if latest_app_id and latest_app_id != str(
                getattr(run_record, "modal_app_id", "") or ""
            ):
                raise RuntimeError(
                    "D1a persisted app binding changed during remote startup"
                )
        function_call_id = str(getattr(latest, "function_call_id", "") or "")
        if function_call_id:
            run_record.function_call_id = function_call_id
            return run_record
        if index + 1 < polls:
            await asyncio.sleep(poll_seconds)
    failure = RuntimeError(
        "D1a FunctionCall ID was not durably joined into the remote TrainingRun"
    )
    if last_transient_error is not None:
        raise failure from last_transient_error
    raise failure


async def _persist_and_verify_d1a_terminal_success(
    run_record: Any,
    *,
    expected_attempt_id: str,
    attempts: int = 3,
    retry_seconds: float = 1.0,
) -> Any:
    """Durably prove D1a success without retrying any scientific work."""

    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("D1a terminal metadata attempts must be a positive integer")
    if (
        isinstance(retry_seconds, bool)
        or not isinstance(retry_seconds, int | float)
        or retry_seconds < 0
    ):
        raise ValueError("D1a terminal metadata retry interval must be nonnegative")
    expected_run_id = str(getattr(run_record, "training_run_id", "") or "")
    expected_app_id = str(getattr(run_record, "modal_app_id", "") or "")
    expected_function_id = str(getattr(run_record, "function_call_id", "") or "")
    expected_root = f"/checkpoints/{expected_run_id}"
    if not all(
        (expected_run_id, expected_app_id, expected_function_id, expected_attempt_id)
    ):
        raise RuntimeError(
            "D1a terminal success lacks its exact run/app/function/attempt IDs"
        )

    record_type = type(run_record)
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            await run_record.save(is_async=True)
            persisted = await record_type.from_id(expected_run_id, is_async=True)
            metadata = getattr(persisted, "metadata", None)
            attempts_ledger = (
                metadata.get("attempts") if isinstance(metadata, dict) else None
            )
            sole_attempt = (
                attempts_ledger[0]
                if isinstance(attempts_ledger, list) and len(attempts_ledger) == 1
                else None
            )
            observed_status = getattr(
                getattr(persisted, "status", None),
                "value",
                getattr(persisted, "status", None),
            )
            if not (
                str(getattr(persisted, "training_run_id", "") or "") == expected_run_id
                and str(getattr(persisted, "modal_app_id", "") or "") == expected_app_id
                and str(getattr(persisted, "function_call_id", "") or "")
                == expected_function_id
                and observed_status == TrainingRunStatus.COMPLETED.value
                and isinstance(metadata, dict)
                and metadata.get("attempt_mode") == "legacy"
                and metadata.get("event_journal_enabled") is True
                and metadata.get("event_journal_contract")
                == "d1a_legacy_single_attempt_v1"
                and metadata.get("attempt_count") == 1
                and metadata.get("active_attempt_id") == expected_attempt_id
                and metadata.get("logical_save_root") == expected_root
                and metadata.get("active_attempt_root") == expected_root
                and isinstance(sole_attempt, dict)
                and sole_attempt.get("attempt") == 1
                and sole_attempt.get("attempt_id") == expected_attempt_id
                and sole_attempt.get("attempt_root") == expected_root
                and sole_attempt.get("status") == "completed"
            ):
                raise RuntimeError(
                    "authoritative D1a terminal run/attempt record differs from success"
                )
            return persisted
        except Exception as exc:
            last_error = exc
            if index + 1 < attempts:
                await asyncio.sleep(retry_seconds)
    raise RuntimeError(
        f"D1a terminal success was not durably persisted after {attempts} metadata attempts"
    ) from last_error


def _restore_d1a_terminal_binding(authoritative: Any, latest: Any) -> Any:
    """Merge the exact live D1a object IDs into a stale mutable cache read."""

    for field in ("modal_app_id", "function_call_id"):
        expected = str(getattr(authoritative, field, "") or "")
        observed = str(getattr(latest, field, "") or "")
        if not expected:
            raise RuntimeError(f"authoritative D1a {field} is empty at terminal save")
        if observed and observed != expected:
            raise RuntimeError(
                f"persisted D1a {field} conflicts with the exact live binding"
            )
        setattr(latest, field, expected)
    expected_url = str(getattr(authoritative, "modal_app_url", "") or "")
    observed_url = str(getattr(latest, "modal_app_url", "") or "")
    if observed_url and expected_url and observed_url != expected_url:
        raise RuntimeError(
            "persisted D1a modal_app_url conflicts with the live binding"
        )
    latest.modal_app_url = expected_url
    return latest


def build_slime_app(
    *,
    training_run_id: str,
    slime: SlimeRecipe,
    model: ModelConfig,
    dataset: DatasetConfig,
    checkpoint: Checkpoint | None = None,
    name: str | None = None,
    group_id: str | None = None,
) -> App:
    """Return a Modal App with `download`, `prepare_dataset`, `convert_checkpoint`, and `train` defined."""
    app_name = name or f"slime-{type(slime).__name__.lstrip('_').lower()}"
    volume_prefix = f"slime-{type(slime).__name__.lstrip('_').lower()}"

    SlimeRecipe._validate_custom_model_architecture(model)
    SlimeRecipe._validate_dataset(dataset)
    _validate_committed_dataset_inputs(slime, dataset)

    # Models that can't do THD packing (model.requires_bshd, e.g. Qwen3-ASR) must
    # train on padded (bshd) batches; fail fast with the fix if the recipe didn't.
    if model and getattr(model, "requires_bshd", False):
        cfg = slime.extra_config or {}
        if cfg.get("qkv_format") != "bshd" or slime.use_dynamic_batch_size:
            raise ValueError(
                f"{model.model_name} requires padded (bshd) batches: its "
                "megatron-bridge forward doesn't implement THD sequence packing. "
                'Set extra_config={"qkv_format": "bshd", "micro_batch_size": N} and '
                "use_dynamic_batch_size=False — or use Qwen3_ASR_1_7b_Recipe, which sets "
                f"these. Got qkv_format={cfg.get('qkv_format')!r}, "
                f"use_dynamic_batch_size={slime.use_dynamic_batch_size}."
            )

    if (
        model
        and getattr(slime, "megatron_to_hf_mode", "") != "bridge"
        and not slime.ref_load
    ):
        # Non-bridge: pre-convert HF -> torch_dist (convert_checkpoint) and load that as the
        # reference checkpoint. In bridge mode we instead load the HF weights directly via
        # AutoBridge; ref_load is set to the local HF snapshot dir at train time.
        slug = model.model_name.replace("/", "--")
        object.__setattr__(slime, "ref_load", f"/checkpoints/torch_dist/{slug}-v31")

    # ── GDN compatibility ─────────────────────────────────────────────────
    # Models with Gated Delta Net (GDN) layers (use_gated_attention=True)
    # don't support packed sequences in the older Megatron-LM bundled with
    # slime.  Slime's get_batch() always creates PackedSeqParams for THD
    # format, which GDN rejects with NotImplementedError.  A build-time
    # patch (patch_gdn_packed_seq.py) neutralises the raise so GDN falls
    # back to unpacked processing.
    _has_gdn = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "use_gated_attention", False)
        and not slime.slime_model_script
    )

    _caller_module, caller_script = resolve_caller_context()

    # ── Image ────────────────────────────────────────────────────────────────
    image = _build_slime_base_image()

    # Hybrid models have layers with different parameter sets (e.g. GDN
    # layers carry linear_attn.dt_bias that standard attention layers lack).
    # Megatron's validate_sharding_integrity rejects this because not every
    # position in the global tensor is covered.  Patch both the conversion
    # and training images so saving and loading both succeed.
    _has_hybrid_spec = (
        model
        and getattr(model, "architecture", None)
        and getattr(model.architecture, "megatron_spec", None)
        and not slime.slime_model_script
    )
    image_overlay = slime.image_overlay
    image_overlay_contract = _image_overlay_contract(
        image_overlay,
        list(slime.image_overlay_source_roots),
        required=slime.attempt_mode == "committed",
    )
    if image_overlay is not None:
        image = image_overlay(image)
        object.__setattr__(slime, "image_overlay", None)

    for patch in slime.patch_files:
        image = image.add_local_file(
            patch,
            remote_path=f"/tmp/{os.path.basename(patch)}",
            copy=True,
        )

    if isinstance(dataset, HarborDataset):
        image = image.uv_pip_install(f"harbor=={HARBOR_PKG_VERSION}")

    if slime.local_slime:
        image = image.add_local_dir(
            slime.local_slime,
            remote_path=SLIME_ROOT,
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv"],
        )

    if slime.image_run_commands:
        image = image.run_commands(*slime.image_run_commands)
    image = _add_training_gym_runtime(image)
    if slime.image_env:
        image = image.env(slime.image_env)

    image = mount_tools_dir(image)

    if caller_script is not None:
        caller_module_name = os.path.splitext(os.path.basename(caller_script))[0]
        caller_remote_path = f"/root/{caller_module_name}.py"
        image = image.add_local_file(
            caller_script,
            remote_path=caller_remote_path,
            copy=True,
        )

    # Patch both conversion and training images for hybrid models.
    # The validation patch lets save/load succeed despite non-uniform
    # layer parameters.  The torch.py patch handles BytesIO entries
    # from _extra_state during checkpoint loading.
    if _has_hybrid_spec:
        image = image.run_commands(
            f"echo {_PATCH_VALIDATION_B64} | base64 -d | python3",
        )

    def _get_custom_generate_path() -> str:
        cfg = slime.extra_config
        if not isinstance(cfg, dict):
            return ""
        raw = cfg.get("custom_generate_function_path", "")
        return raw if isinstance(raw, str) else ""

    def _set_custom_generate_path(path: str) -> None:
        cfg = dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
        cfg["custom_generate_function_path"] = path
        object.__setattr__(slime, "extra_config", cfg)

    def _set_extra_config_path(key: str) -> Callable[[str], None]:
        def setter(path: str) -> None:
            cfg = (
                dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
            )
            cfg[key] = path
            object.__setattr__(slime, "extra_config", cfg)

        return setter

    def _ship_callable(
        fn: Any,
        *,
        fallback_name: str,
        set_path: Callable[[str], None],
    ) -> None:
        nonlocal image
        image = ship_callable(
            image,
            fn,
            caller_script=caller_script,
            fallback_name=fallback_name,
            set_path=set_path,
        )

    def _set_custom_rm_path(path: str) -> None:
        cfg = dict(slime.extra_config) if isinstance(slime.extra_config, dict) else {}
        cfg["custom_rm_path"] = path
        object.__setattr__(slime, "extra_config", cfg)

    _ship_callable(
        slime.custom_rm_function,
        fallback_name="custom_rm",
        set_path=_set_custom_rm_path,
    )
    _ship_callable(
        slime.custom_generate_function,
        fallback_name="custom_generate",
        set_path=_set_custom_generate_path,
    )
    _ship_callable(
        slime.rollout_function if callable(slime.rollout_function) else None,
        fallback_name="rollout_function",
        set_path=lambda path: object.__setattr__(slime, "rollout_function", path),
    )
    for attr, config_key, fallback_name in (
        (
            "custom_rollout_log_function",
            "training_gym_custom_rollout_log_function_path",
            "custom_rollout_log",
        ),
        (
            "custom_eval_rollout_log_function",
            "training_gym_custom_eval_rollout_log_function_path",
            "custom_eval_rollout_log",
        ),
        (
            "custom_megatron_before_log_prob_hook",
            "training_gym_custom_megatron_before_log_prob_hook_path",
            "before_log_prob_hook",
        ),
        (
            "custom_megatron_before_train_step_hook",
            "training_gym_custom_megatron_before_train_step_hook_path",
            "before_train_step_hook",
        ),
    ):
        value = getattr(slime, attr)
        _ship_callable(
            value if callable(value) else None,
            fallback_name=fallback_name,
            set_path=_set_extra_config_path(config_key),
        )
        if callable(value):
            object.__setattr__(slime, attr, None)

    if slime.custom_rm_function is not None:
        object.__setattr__(slime, "custom_rm_function", None)
    if slime.custom_generate_function is not None and _get_custom_generate_path():
        object.__setattr__(slime, "custom_generate_function", None)

    # ── SGLang request params auto-wiring ─────────────────────────────────
    if slime.sglang_request_params:
        cfg = dict(slime.extra_config or {})
        cfg["sglang_request_params"] = slime.sglang_request_params
        if "custom_rm_path" not in cfg:
            cfg["custom_rm_path"] = (
                "modal_training_gym.frameworks.slime.opd_reward.reward_func"
            )
        if "custom_reward_post_process_path" not in cfg:
            cfg["custom_reward_post_process_path"] = (
                "modal_training_gym.frameworks.slime.opd_reward.post_process_rewards"
            )
        object.__setattr__(slime, "extra_config", cfg)

    # Build train_image AFTER _ship_callable so shipped modules are included.
    train_image = image
    if _has_hybrid_spec:
        train_image = image.run_commands(
            f"echo {_PATCH_TORCH_LOAD_B64} | base64 -d | python3",
            f"echo {_PATCH_GLOBAL_PLAN_B64} | base64 -d | python3",
            f"echo {_PATCH_CHECKPOINT_SAVE_B64} | base64 -d | python3",
        )
    if _has_gdn:
        train_image = train_image.run_commands(
            f"echo {_PATCH_GDN_PACKED_SEQ_B64} | base64 -d | python3",
        )
    if slime.megatron_to_hf_mode == "bridge":
        train_image = train_image.run_commands(
            f"echo {_PATCH_BRIDGE_PER_TOKEN_LOSS_B64} | base64 -d | python3",
        )

    # ── Volumes ──────────────────────────────────────────────────────────────
    hf_cache_volume = Volume.from_name("huggingface-cache", create_if_missing=True)
    data_volume = Volume.from_name(f"{volume_prefix}-data", create_if_missing=True)
    checkpoints_volume_name, checkpoints_mount_path, checkpoints_volume = (
        resolve_checkpoint_volumes(
            checkpoint,
            volume_prefix=volume_prefix,
            default_mount_path=str(CHECKPOINTS_PATH),
        )
    )
    metadata_volume = Volume.from_name("training-gym-metadata", create_if_missing=True)
    if checkpoint is not None and checkpoint.path and not model.model_path:
        model.model_path = checkpoint.path
    all_volumes: dict[str | PurePosixPath, Any] = {
        str(HF_CACHE_PATH): hf_cache_volume,
        str(DATA_PATH): data_volume,
        checkpoints_mount_path: checkpoints_volume,
        "/metadata": metadata_volume,
    }

    # ── App ──────────────────────────────────────────────────────────────────
    tags = build_app_tags(
        framework="slime",
        model=model,
        recipe_app_tags=slime.app_tags,
        wandb=slime.wandb,
    )
    app = App(app_name, tags=tags)
    gpu_spec = f"{slime.gpu_type}:{slime.gpu_allocation.gpus_per_node}"
    conversion_gpus = (
        slime.actor_num_gpus_per_node
        if slime.conversion_gpus_per_node is None
        else slime.conversion_gpus_per_node
    )
    conversion_gpu_spec = f"{slime.gpu_type}:{conversion_gpus}"

    @app.function(
        image=image,
        volumes={
            str(HF_CACHE_PATH): hf_cache_volume,
            checkpoints_mount_path: checkpoints_volume,
        },
        timeout=6 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="download",
    )
    def download(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        run_download_phase(
            training_run_id=training_run_id,
            phase=SlimeStatus.DOWNLOAD_MODEL.value,
            framework_status_url=framework_status_url,
            framework_status_token=framework_status_token,
            volumes=(hf_cache_volume, checkpoints_volume),
            download=model.download,
        )

    @app.function(
        image=image,
        volumes={str(DATA_PATH): data_volume},
        timeout=2 * 60 * 60,
        secrets=hf_secrets(),
        serialized=True,
        name="prepare_dataset",
    )
    def prepare_dataset():
        run_prepare_dataset(dataset, data_volume, SlimeRecipe._resolve_data_paths)

    convert_nnodes = get_checkpoint_conversion_policy(slime, model=model)[0]

    @app.function(
        image=image,
        gpu=conversion_gpu_spec,
        memory=slime.memory,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        timeout=4 * 60 * 60,
        experimental_options={"efa_enabled": True},
        serialized=True,
        name="convert_checkpoint",
    )
    @clustered_if(convert_nnodes > 1, convert_nnodes, gpu_type=slime.gpu_type)
    def convert_checkpoint(
        training_run_id: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        from huggingface_hub import snapshot_download
        from modal_training_gym.common.status_reporter import (
            enqueue_framework_status,
            flush as flush_status_reporter,
        )

        if training_run_id:
            enqueue_framework_status(
                training_run_id,
                SlimeStatus.CONVERT_MODEL.value,
                url=framework_status_url or None,
                token=framework_status_token or None,
                is_active=True,
            )

        # Bridge mode loads the HF weights directly into Megatron via AutoBridge at train time
        # (slime's _load_checkpoint_hf), so there is no offline HF→torch_dist conversion to run.
        # The HF reference path is wired into `ref_load` in `train` below.
        if getattr(slime, "megatron_to_hf_mode", None) == "bridge":
            print(
                "Bridge mode — HF weights loaded directly via AutoBridge; no conversion needed."
            )
            if training_run_id:
                flush_status_reporter(timeout_seconds=2.0)
            return

        hf_cache_volume.reload()
        checkpoints_volume.reload()

        if slime.megatron_conversion_hf_checkpoint:
            hf_path = resolve_checkpoint_ref(slime.megatron_conversion_hf_checkpoint)
        elif model.model_path:
            hf_path = str(model.model_path)
        else:
            hf_path = snapshot_download(model.model_name, local_files_only=True)
        save_path = str(slime.ref_load)

        num_nodes, nproc_per_node, extra_args = get_checkpoint_conversion_policy(
            slime, model=model
        )
        node_rank, master_addr, _, nnodes = get_modal_cluster_context(num_nodes)

        import json
        import shutil

        current_config = _build_conversion_config(slime, model=model)

        if os.path.exists(save_path):
            complete = has_torch_dist_checkpoint(
                save_path, is_complete=_is_complete_torch_dist_checkpoint
            )
            stale = True
            if complete:
                config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
                if os.path.isfile(config_path):
                    try:
                        with open(config_path) as f:
                            stored_config = json.load(f)
                        stale = stored_config != current_config
                        if stale and node_rank == 0:
                            print(
                                f"Checkpoint at {save_path} was built with "
                                f"different config:\n  stored: {stored_config}"
                                f"\n  current: {current_config}"
                            )
                    except (OSError, json.JSONDecodeError):
                        stale = True
                        if node_rank == 0:
                            print(
                                f"Checkpoint at {save_path} has unreadable "
                                f"conversion config — reconverting."
                            )
                else:
                    # Legacy checkpoint without config metadata — cannot
                    # verify compatibility.  Force re-conversion so the
                    # checkpoint is rebuilt with the correct layout and
                    # metadata for future validation.
                    stale = True
                    if node_rank == 0:
                        print(
                            f"Checkpoint at {save_path} has no conversion "
                            f"config metadata — reconverting to ensure "
                            f"compatibility."
                        )
                if not stale:
                    if node_rank == 0:
                        print(f"Using existing torch_dist checkpoint at {save_path}.")
                    if training_run_id:
                        flush_status_reporter(timeout_seconds=2.0)
                    return

            # Either incomplete (e.g. a preempted conversion) or stale: rebuild
            # it. Rank 0 removes the directory and commits; other ranks wait for
            # the removal to land before reconverting.
            if node_rank == 0:
                import shutil

                reason = "stale" if complete else "incomplete"
                print(f"Removing {reason} torch_dist checkpoint at {save_path}.")
                shutil.rmtree(save_path, ignore_errors=True)
                checkpoints_volume.commit()
            else:
                time.sleep(5)
                checkpoints_volume.reload()

        torchrun_args = [f"--nproc-per-node={nproc_per_node}"]
        if nnodes > 1:
            torchrun_args += [
                f"--nnodes={nnodes}",
                f"--node-rank={node_rank}",
                f"--master-addr={master_addr}",
                "--master-port=12355",
            ]

        import importlib.util

        mmt = ""
        if model and getattr(model, "architecture", None):
            mmt = getattr(model.architecture, "megatron_model_type", "")

        if num_nodes > 1:
            spec = importlib.util.find_spec(
                "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist"
            )
            convert_script = spec.origin if spec is not None else None
            if not convert_script:
                raise RuntimeError(
                    "modal_training_gym.frameworks.slime.modal_helpers.convert_hf_to_torch_dist not found"
                )
        else:
            convert_script = f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
        if mmt or slime.slime_model_script:
            model_script = (
                f"{SLIME_ROOT}/{slime.slime_model_script}"
                if slime.slime_model_script
                else f"{SLIME_ROOT}/scripts/models/{mmt}.sh"
            )
            cmd = (
                f"source {model_script} && "
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                '"${MODEL_ARGS[@]}" '
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )
        else:
            cmd = (
                f"torchrun {' '.join(torchrun_args)} {convert_script} "
                f"{' '.join(extra_args)} "
                f"--hf-checkpoint {shlex.quote(hf_path)} --save {shlex.quote(save_path)}"
            )

        env = {**os.environ, **slime.environment}
        env.pop("NCCL_NVLS_ENABLE", None)
        if num_nodes > 1:
            env["SKIP_RELEASE_RENAME"] = "1"
        print(
            f"Conversion layout: nodes={num_nodes}, "
            f"nproc_per_node={nproc_per_node}, node_rank={node_rank}"
        )
        print(f"Running: bash -c {cmd!r}")
        subprocess.run(["bash", "-c", cmd], check=True, env=env)

        if node_rank == 0:
            config_path = os.path.join(save_path, _CONVERSION_CONFIG_FILE)
            try:
                with open(config_path, "w") as f:
                    json.dump(current_config, f)
            except OSError as exc:
                print(f"WARNING: could not write conversion config: {exc}")
        checkpoints_volume.commit()

        if node_rank == 0:
            print(f"Saved torch_dist checkpoint to {save_path}")

        if training_run_id:
            flush_status_reporter(timeout_seconds=2.0)

    # Use Modal's clustered scheduler with RDMA when using a full node (8+ GPUs)
    # on RDMA-capable hardware, or for any multi-node run.  The `rdma=True` flag
    # provides CAP_IPC_LOCK and NVSwitch device access that slime's colocated
    # weight sync (UpdateWeightFromTensor) needs for fast CUDA IPC transfers.
    _multi_node = slime.total_nodes > 1
    _full_node = slime.gpu_allocation.gpus_per_node >= 8
    _use_clustered = _multi_node or (_full_node and _supports_rdma(slime.gpu_type))

    train_secrets: list[Secret] = []
    if slime.wandb is not None:
        train_secrets.append(Secret.from_name(slime.wandb.modal_wandb_secret_name))
    train_experimental_options: dict[str, Any] = {"efa_enabled": True}

    train_function_kwargs = dict(slime.train_function_kwargs or {})
    user_secrets = train_function_kwargs.pop("secrets", None)
    if user_secrets is not None:
        if not isinstance(user_secrets, (list, tuple)):
            user_secrets = [user_secrets]
        train_secrets.extend(user_secrets)
    user_experimental_options = train_function_kwargs.pop("experimental_options", None)
    if user_experimental_options is not None:
        train_experimental_options.update(user_experimental_options)
    train_ephemeral_disk = train_function_kwargs.pop("ephemeral_disk", None)
    train_timeout_seconds = _pop_train_function_timeout(train_function_kwargs)
    if train_function_kwargs:
        unsupported = ", ".join(sorted(train_function_kwargs))
        raise TypeError(f"Unsupported slime.train_function_kwargs keys: {unsupported}")

    scientific_run_contract = (
        _scientific_run_contract(
            training_run_id=training_run_id,
            recipe=slime,
            model=model,
            dataset=dataset,
            checkpoint=checkpoint,
            caller_script=caller_script,
            has_hybrid_spec=bool(_has_hybrid_spec),
            has_gdn=bool(_has_gdn),
            train_ephemeral_disk=train_ephemeral_disk,
            train_timeout_seconds=train_timeout_seconds,
            train_experimental_options=train_experimental_options,
            image_overlay_contract=image_overlay_contract,
        )
        if slime.attempt_mode == "committed"
        else None
    )
    scientific_run_contract_sha256 = (
        run_contract_sha256(scientific_run_contract)
        if scientific_run_contract is not None
        else ""
    )

    async def write_step_times(
        run_id: str,
        attempt_id: str,
        num_steps: int,
    ) -> dict[str, dict[str, int | None]]:
        step_times_dict = ModalDict.from_name(
            "training-gym-step-times", create_if_missing=True
        )

        step_times: dict[str, dict[str, int | None]] = {}
        for current_step_num in range(1, num_steps + 1):
            start_key = f"{run_id}:{attempt_id}:{current_step_num}:start"
            finish_key = f"{run_id}:{attempt_id}:{current_step_num}:finish"

            current_step_start_time, current_step_end_time = await asyncio.gather(
                step_times_dict.get.aio(start_key),
                step_times_dict.get.aio(finish_key),
            )
            if current_step_start_time is not None:
                current_step_start_time = int(current_step_start_time)
            if current_step_end_time is not None:
                current_step_end_time = int(current_step_end_time)

            duration = None
            if (
                current_step_start_time is not None
                and current_step_end_time is not None
            ):
                duration = current_step_end_time - current_step_start_time

            step_times[f"{current_step_num}"] = {
                "start": current_step_start_time,
                "end": current_step_end_time,
                "duration_s": duration,
            }

        return step_times

    async def clear_step_times(
        run_id: str,
        attempt_id: str,
        num_steps: int,
    ) -> None:
        step_times_dict = ModalDict.from_name(
            "training-gym-step-times", create_if_missing=True
        )

        await asyncio.gather(
            *(
                step_times_dict.pop.aio(key, None)
                for current_step_num in range(1, num_steps + 1)
                for key in (
                    f"{run_id}:{attempt_id}:{current_step_num}:start",
                    f"{run_id}:{attempt_id}:{current_step_num}:finish",
                )
            )
        )

    @app.function(
        image=train_image,
        gpu=gpu_spec,
        memory=slime.memory,
        cloud=slime.cloud,
        region=slime.region,
        volumes=all_volumes,
        secrets=train_secrets or None,
        ephemeral_disk=train_ephemeral_disk,
        timeout=train_timeout_seconds,
        # Retry policy is recipe-controlled. In committed attempt mode a retry
        # writes to a fresh namespace and may load only an authenticated boundary;
        # without a boundary it restarts from the recipe's original initialization.
        # ``Retries(max_retries=0)`` is needlessly ambiguous at the protobuf
        # boundary because zero-valued scalar fields are elided.  Use Modal's
        # no-policy default for the zero case; positive values retain the
        # explicit recipe-controlled policy.
        retries=_modal_retry_policy(slime.max_retries),
        single_use_containers=True,
        experimental_options=train_experimental_options or None,
        serialized=True,
        name="train",
    )
    @clustered_if(_use_clustered, slime.total_nodes, gpu_type=slime.gpu_type)
    async def train(
        modal_app_id: str = "",
        modal_app_url: str = "",
        framework_status_url: str = "",
        framework_status_token: str = "",
    ):
        # Modal's native timeout begins at this function entry.  Capture one
        # authoritative wall/monotonic pair before cluster discovery, volume
        # reloads, attempt creation, Ray bootstrap, or model setup, and pass it
        # unchanged to the receipt-bound worker runtime.
        remote_entry_clock = _capture_remote_entry_clock(slime.environment or {})
        modal_app_id = modal_app_id or os.environ.get("MODAL_APP_ID", "")
        remote_execution_identity = _remote_execution_identity(modal_app_id)
        modal_app_url = modal_app_url or modal_app_dashboard_url(modal_app_id)

        # Make the dashboard URL visible to both the launcher's own
        # status_reporter and (via runtime_env below) the slime worker
        # process. The toml file lives on the user's local machine and isn't
        # accessible inside this container, so the URL has to be passed in.
        if framework_status_url:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_URL"] = framework_status_url
        if framework_status_token:
            os.environ["TRAINING_GYM_FRAMEWORK_STATUS_TOKEN"] = framework_status_token

        setup_rank_is_head: bool | None = None

        async def _initialize_cluster_and_attempt():
            nonlocal setup_rank_is_head
            initialized_cluster = ModalRayCluster()
            initialized_cluster.discover_cluster(slime.total_nodes)
            setup_rank_is_head = initialized_cluster.is_head
            initialized_cluster.emit_member_identity(training_run_id=training_run_id)

            await asyncio.gather(
                hf_cache_volume.reload.aio(),
                data_volume.reload.aio(),
                checkpoints_volume.reload.aio(),
            )

            if slime.attempt_mode == "committed":
                accepted_result = await load_accepted_train_result(
                    training_run_id,
                    expected_framework=Framework.SLIME,
                    expected_run_contract_sha256=scientific_run_contract_sha256,
                )
                if accepted_result is not None:
                    print(
                        "Authenticated accepted TrainResult already exists; "
                        "returning it without creating another logical attempt",
                        flush=True,
                    )
                    return _AcceptedTrainResultReplay(
                        accepted_result._to_dict()
                        if initialized_cluster.is_head
                        else None
                    )

            os.environ["SLIME_HOST_IP"] = initialized_cluster.node_ip
            os.environ["SGLANG_HOST_IP"] = initialized_cluster.node_ip
            os.environ["HOST_IP"] = initialized_cluster.node_ip
            # Modal injects the W&B Secret independently into every clustered
            # container. Install the backwards-compatible config fallback
            # before Ray starts as well; Ray workers then inherit authentication
            # without the key appearing in JobSubmission runtime_env metadata.
            install_wandb_api_key_in_process(slime.wandb)

            initialized_wait_retries = int(
                (slime.environment or {}).get(
                    "TRAINING_GYM_RAY_WORKER_WAIT_RETRIES",
                    os.environ.get("TRAINING_GYM_RAY_WORKER_WAIT_RETRIES", "60"),
                )
            )
            if not initialized_cluster.is_head:
                initialized_cluster.start_ray(
                    worker_wait_retries=initialized_wait_retries
                )

                accepted_completion_probe = None
                if slime.attempt_mode == "committed":

                    async def _accepted_completion_probe() -> bool:
                        return (
                            await load_accepted_train_result(
                                training_run_id,
                                expected_framework=Framework.SLIME,
                                expected_run_contract_sha256=(
                                    scientific_run_contract_sha256
                                ),
                            )
                            is not None
                        )

                    accepted_completion_probe = _accepted_completion_probe

                await initialized_cluster.wait_forever(
                    accepted_completion_probe=accepted_completion_probe
                )
                return None

            # Create the logical attempt record before the Ray head waits for
            # every worker. Otherwise a missing second node exits before any
            # structured record exists.
            initialized_wandb_entity = (
                slime.wandb.entity or os.environ.get("WANDB_ENTITY", "")
                if slime.wandb is not None
                else ""
            )
            initialized_wandb_run_id = ""

            print(f"Training run id: {training_run_id}")
            config_summary: dict = {
                "model": {"model_name": model.model_name} if model else {},
                "recipe": _serialize_slime_params(
                    slime,
                    dataset=dataset,
                    model=model,
                ),
                "wandb": (
                    {
                        "project": slime.wandb.project,
                        "group": slime.wandb.group,
                        "entity": initialized_wandb_entity,
                        "run_id": initialized_wandb_run_id,
                    }
                    if slime.wandb
                    else {}
                ),
                "dataset": {
                    "hf_repo": getattr(dataset, "hf_repo", ""),
                    "name": type(dataset).__name__,
                },
                "lr": slime.lr,
                "global_batch_size": slime.global_batch_size,
                "scientific_run_contract_sha256": scientific_run_contract_sha256,
                "scientific_run_contract": scientific_run_contract,
            }
            (
                initialized_run_record,
                initialized_wandb_run_id,
                initialized_status_token,
            ) = await init_training_run_record(
                training_run_id=training_run_id,
                modal_app_id=modal_app_id,
                modal_app_url=(modal_app_url or modal_app_dashboard_url(modal_app_id)),
                framework=Framework.SLIME,
                initializing_status=SlimeStatus.INITIALIZING,
                config_summary=config_summary,
                wandb_cfg=slime.wandb,
                wandb_entity=initialized_wandb_entity,
                framework_status_token=framework_status_token,
                max_attempts=slime.max_attempts,
            )
            initialized_run_record = await _require_d1a_function_call_binding(
                initialized_run_record
            )
            return (
                initialized_cluster,
                initialized_wait_retries,
                initialized_run_record,
                initialized_wandb_entity,
                initialized_wandb_run_id,
                initialized_status_token,
            )

        try:
            initialized = await _initialize_cluster_and_attempt()
        except BaseException as exc:
            if isinstance(exc, AcceptedTrainResultError):
                # Authentication failed before attempt creation. Preserve the
                # already-published logical-run record and artifacts verbatim.
                raise
            # Only rank 0 owns logical-run state. Worker cancellation and
            # worker-local setup failures are evidence for the head/Ray/Modal
            # diagnostics, not authority to terminalize the shared attempt.
            if not _setup_rank_owns_logical_run_failure(setup_rank_is_head):
                raise
            primary_error = await record_setup_failure(training_run_id, exc)
            current_error = f"{type(exc).__name__}: {exc}"
            if primary_error and primary_error != current_error:
                print(
                    "Retry setup failed after the causal failure; surfacing "
                    f"the preserved primary failure: {primary_error}",
                    flush=True,
                )
                raise RuntimeError(primary_error) from exc
            raise
        if isinstance(initialized, _AcceptedTrainResultReplay):
            return initialized.payload
        if initialized is None:
            return
        (
            cluster,
            ray_worker_wait_retries,
            run_record,
            wandb_entity,
            wandb_run_id,
            framework_status_token,
        ) = initialized
        attempt_metadata = dict(run_record.metadata or {})
        attempt_id = str(attempt_metadata.get("active_attempt_id") or "")
        attempt_count = int(attempt_metadata.get("attempt_count") or 0)
        ray_started = False
        ray_diagnostic_recorded = False
        ray_failure_stage = "ray_cluster_bootstrap"
        committed_attempt_mode = slime.attempt_mode == "committed"
        logical_save_root: str | None = None

        try:  # Wraps all post-setup work so any failure marks the run terminal.
            _validate_attempt_limit(attempt_count, slime.max_attempts)
            record_training_attempt_cluster_identity(
                run_record, cluster.identity_snapshot()
            )
            # Persist the attempt-to-Modal-cluster mapping before Ray bootstrap;
            # a worker can disappear while the head is waiting for Ray nodes.
            await run_record.save(is_async=True)
            cluster.start_ray(worker_wait_retries=ray_worker_wait_retries)
            ray_started = True
            ray_failure_stage = "post_ray_start_setup"

            # Fail fast on W&B access before model initialization or a Ray job,
            # rather than surfacing a recurring CommError mid-training.
            if slime.wandb is not None:
                wandb_entity = _preflight_wandb(slime.wandb)
                run_record.config["wandb"]["entity"] = wandb_entity
                record_wandb_attempt(
                    run_record,
                    entity=wandb_entity,
                    project=slime.wandb.project,
                    group=slime.wandb.group,
                    run_id=wandb_run_id,
                    attempt_count=attempt_count,
                )

            # In-flight status updates are fire-and-forget via the dashboard's
            # /api/framework-status endpoint so the training thread doesn't pay
            # the ~300ms volume-write latency on each transition. Terminal state
            # (COMPLETED/FAILED/STOPPED) still goes through
            # run_record.save(is_async=True) below to guarantee delivery
            # before the container exits.
            from modal_training_gym.common.status_reporter import (
                enqueue_framework_status,
            )

            def _set_framework_status(status: SlimeStatus) -> None:
                run_record.framework_status = status
                enqueue_framework_status(
                    training_run_id,
                    status.value,
                    token=framework_status_token,
                    attempt_id=attempt_id,
                )

            async def _set_framework_status_async(status: SlimeStatus) -> None:
                _set_framework_status(status)

            if model:
                await _set_framework_status_async(SlimeStatus.DOWNLOAD_MODEL)
                cache_dir = (
                    HF_CACHE_PATH
                    / "hub"
                    / f"models--{model.model_name.replace('/', '--')}"
                )
                snapshots_dir = cache_dir / "snapshots"
                has_snapshot = snapshots_dir.is_dir() and any(snapshots_dir.iterdir())
                if not has_snapshot:
                    print(f"Downloading model {model.model_name}...")
                model.download()  # Always run (idempotent; applies config patches to cached snapshots)
                await hf_cache_volume.commit.aio()

            if dataset:
                await _set_framework_status_async(SlimeStatus.PREPARE_DATASET)
                prompt_data, eval_paths = SlimeRecipe._resolve_data_paths(dataset)
                needs_prepare = not os.path.exists(prompt_data)
                if dataset.always_prepare and os.path.exists(prompt_data):
                    import shutil

                    data_dir = os.path.dirname(prompt_data)
                    print(f"always_prepare=True — removing {data_dir}")
                    shutil.rmtree(data_dir, ignore_errors=True)
                    needs_prepare = True
                if needs_prepare:
                    print(f"Preparing dataset ({prompt_data})...")
                    dataset.prepare(prompt_data, eval_paths)
                    await data_volume.commit.aio()
                dataset.validate_prepared(prompt_data)
                for ep in (eval_paths or {}).values():
                    if os.path.exists(ep):
                        dataset.validate_prepared(ep)

            await _set_framework_status_async(SlimeStatus.CONVERT_MODEL)

            logical_save_root = compute_save_root(
                slime.save,
                recipe_default_save_root=str(CHECKPOINTS_PATH).rstrip("/"),
                mounted_save_root=checkpoints_mount_path,
                training_run_id=training_run_id,
            )
            if committed_attempt_mode:
                mounted_root = Path(checkpoints_mount_path).resolve()
                resolved_logical_root = Path(logical_save_root).resolve()
                if (
                    resolved_logical_root != mounted_root
                    and mounted_root not in resolved_logical_root.parents
                ):
                    raise RuntimeError(
                        "committed attempt mode requires the logical save root "
                        "to be inside the mounted checkpoints Volume: "
                        f"save={resolved_logical_root} mount={mounted_root}"
                    )
            resume_boundary: dict[str, Any] | None = None
            terminal_parent_complete = False
            if committed_attempt_mode:
                if not attempt_id or attempt_count < 1:
                    raise RuntimeError(
                        "committed attempt mode requires an initialized attempt identity"
                    )
                resume_boundary = load_latest_committed_boundary(
                    logical_save_root,
                    verify_hashes=True,
                    expected_run_contract_sha256=scientific_run_contract_sha256,
                )
                if resume_boundary is not None and resume_boundary["terminal"]:
                    expected_terminal_rollout = slime.num_rollout - 1
                    if int(resume_boundary["rollout_id"]) != expected_terminal_rollout:
                        raise RuntimeError(
                            "terminal committed boundary does not match this recipe: "
                            f"boundary={resume_boundary['rollout_id']} "
                            f"expected={expected_terminal_rollout}"
                        )
                    terminal_parent_complete = True
                    accepted_wandb_run_id = select_accepted_wandb_attempt(
                        run_record,
                        accepted_attempt_id=str(resume_boundary["attempt_id"]),
                        skipped_attempt_count=attempt_count,
                    )
                    if accepted_wandb_run_id:
                        wandb_run_id = accepted_wandb_run_id
                        if slime.wandb is not None:
                            run_record.config["wandb"]["run_id"] = wandb_run_id
                initial_attempt_load = (
                    str(
                        attempt_root(
                            logical_save_root,
                            resume_boundary["attempt_id"],
                        )
                    )
                    if resume_boundary is not None
                    else str(slime.load or slime.ref_load or "")
                )
                save_root = str(
                    create_attempt_namespace(
                        logical_save_root,
                        run_id=training_run_id,
                        attempt_id=attempt_id,
                        attempt_count=attempt_count,
                        initial_load=initial_attempt_load,
                        parent_boundary=resume_boundary,
                        run_contract_sha256=scientific_run_contract_sha256,
                    )
                )
                # Worker-node containers mounted this Volume before the head
                # allocated the attempt. Persist the immutable owner now; the
                # framework's startup barrier reloads it on every writer node.
                await checkpoints_volume.commit.aio()
            else:
                save_root = logical_save_root

            original_save = slime.save
            original_load = slime.load
            original_ref_load = slime.ref_load
            original_extra_config = slime.extra_config
            object.__setattr__(slime, "save", save_root)

            if committed_attempt_mode:
                if resume_boundary is None:
                    resume_checkpoint = None
                else:
                    checkpoint_path = str(
                        Path(logical_save_root)
                        / str(resume_boundary["checkpoint_path"])
                    )
                    if not _is_complete_torch_dist_checkpoint(checkpoint_path):
                        raise RuntimeError(
                            "committed resume boundary references an incomplete "
                            f"checkpoint: {checkpoint_path}"
                        )
                    resume_checkpoint = {
                        "resume_checkpoint_path": checkpoint_path,
                        "resume_checkpoint_name": Path(checkpoint_path).name,
                        "resume_from_iteration": int(
                            resume_boundary["checkpoint_iteration"]
                        ),
                    }
            else:
                resume_checkpoint = torch_dist_resume_checkpoint(
                    save_root, is_complete=_is_complete_torch_dist_checkpoint
                )
            record_resume_checkpoint(run_record, resume_checkpoint)
            run_metadata = dict(run_record.metadata or {})
            run_metadata.update(
                {
                    "active_attempt_id": attempt_id,
                    "active_attempt_root": save_root,
                    "logical_save_root": logical_save_root,
                    "attempt_mode": slime.attempt_mode,
                    "max_retries": slime.max_retries,
                    "scientific_run_contract_sha256": scientific_run_contract_sha256,
                    "resume_boundary": (
                        {
                            key: resume_boundary[key]
                            for key in (
                                "attempt_id",
                                "rollout_id",
                                "checkpoint_iteration",
                                "terminal",
                                "boundary_manifest",
                                "boundary_sha256",
                            )
                        }
                        if resume_boundary is not None
                        else None
                    ),
                    "finalized_from_terminal_parent": terminal_parent_complete,
                }
            )
            raw_attempts = run_metadata.get("attempts")
            if isinstance(raw_attempts, list):
                updated_attempts: list[Any] = []
                for raw_attempt in raw_attempts:
                    if (
                        isinstance(raw_attempt, dict)
                        and raw_attempt.get("attempt_id") == attempt_id
                    ):
                        attempt_record = dict(raw_attempt)
                        attempt_record.update(
                            {
                                "attempt_root": save_root,
                                "resume_from": run_metadata["resume_boundary"],
                            }
                        )
                        updated_attempts.append(attempt_record)
                    else:
                        updated_attempts.append(raw_attempt)
                run_metadata["attempts"] = updated_attempts
            run_record.metadata = run_metadata
            await run_record.save(is_async=True)

            try:
                if resume_checkpoint is not None:
                    print(
                        f"WARNING: detected existing checkpoint in "
                        f"{resume_checkpoint['resume_checkpoint_path']}; "
                        "resuming training from last saved iteration."
                    )
                    if committed_attempt_mode:
                        assert resume_boundary is not None
                        parent_attempt_root = str(
                            attempt_root(
                                logical_save_root,
                                resume_boundary["attempt_id"],
                            )
                        )
                        object.__setattr__(slime, "load", parent_attempt_root)
                        # This must happen before prepare_slime_config materializes
                        # extra_config into YAML. Updating it afterwards either
                        # fails on the path string or silently omits --ckpt-step.
                        resume_extra_config = dict(slime.extra_config or {})
                        resume_extra_config["ckpt_step"] = int(
                            resume_boundary["checkpoint_iteration"]
                        )
                        object.__setattr__(slime, "extra_config", resume_extra_config)
                    else:
                        object.__setattr__(slime, "load", save_root)

                prepare_slime_config(slime, model, tempfile.mkdtemp())

                # Resolve the local HF snapshot dir (used for a fresh bridge-mode
                # load below). prepare_slime_config may have populated model_path.
                _hf_ref: str | None = None
                if model and (slime.megatron_to_hf_mode == "bridge" or slime.ref_load):
                    from huggingface_hub import snapshot_download as _snap0

                    _hf_ref = (
                        str(model.model_path)
                        if model.model_path
                        else _snap0(model.model_name, local_files_only=True)
                    )

                if (
                    resume_checkpoint is None
                    and slime.megatron_to_hf_mode == "bridge"
                    and not slime.ref_load
                    and _hf_ref
                ):
                    # Fresh bridge run: load HF weights via AutoBridge. Pointing
                    # ref_load at torch_dist would trigger full-resume semantics
                    # and require optimizer/RNG state.
                    object.__setattr__(slime, "ref_load", _hf_ref)
                cmd = build_train_cmd(slime, SLIME_ROOT, model=model, dataset=dataset)
            finally:
                object.__setattr__(slime, "save", original_save)
                object.__setattr__(slime, "load", original_load)
                object.__setattr__(slime, "ref_load", original_ref_load)
                object.__setattr__(slime, "extra_config", original_extra_config)

            phase_report_url = (
                os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL")
                or framework_status_url
                or ""
            )
            if not phase_report_url:
                print(
                    "WARNING: no dashboard URL passed to train() and no "
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL set inside the "
                    "container. Phase reporting is disabled for this run."
                )

            wandb_env = build_wandb_runtime_env(
                slime.wandb,
                run_id=wandb_run_id,
                entity=wandb_entity,
            )

            runtime_env = {
                "env_vars": {
                    "no_proxy": f"127.0.0.1,{cluster.head_addr}",
                    "MASTER_ADDR": cluster.head_addr,
                    "TRAINING_GYM_TRAINING_RUN_ID": training_run_id,
                    "TRAINING_GYM_APP_NAME": app_name,
                    **remote_execution_identity,
                    "TRAINING_GYM_TOTAL_STEPS": str(slime.num_rollout),
                    "TRAINING_GYM_RESPONSE_PARSER_PATH": _response_parser_path(model),
                    "TRAINING_GYM_CAPTURE_TRACE": (
                        "1" if getattr(slime, "capture_trace", False) else ""
                    ),
                    "TRAINING_GYM_TRACE_SAMPLE_LIMIT": str(
                        getattr(slime, "trace_sample_limit", 16)
                    ),
                    "TRAINING_GYM_IMAGE_SAMPLE_LIMIT": str(
                        getattr(slime, "image_sample_limit", 16)
                    ),
                    "TRAINING_GYM_TRAJECTORY_SAMPLE_LIMIT": str(
                        getattr(slime, "trajectory_sample_limit", 16)
                    ),
                    "TRAINING_GYM_FRAMEWORK_STATUS_URL": phase_report_url,
                    **wandb_env,
                    **_remote_entry_runtime_env(
                        slime.environment or {},
                        remote_entry_clock,
                    ),
                    "TRAINING_GYM_ATTEMPT_MODE": slime.attempt_mode,
                    "TRAINING_GYM_ATTEMPT_ID": attempt_id,
                    "TRAINING_GYM_LOGICAL_SAVE_ROOT": logical_save_root,
                    "TRAINING_GYM_CHECKPOINTS_VOLUME_NAME": checkpoints_volume_name,
                    "TRAINING_GYM_RUN_CONTRACT_SHA256": (
                        scientific_run_contract_sha256
                    ),
                    "DRIFT_ASYNC_RL_ATTEMPT_ID": attempt_id,
                    "DRIFT_ASYNC_RL_WRITER_ATTEMPT_ID": attempt_id,
                    "DRIFT_ASYNC_RL_GENERATION_ATTEMPT_ID": attempt_id,
                    "DRIFT_ASYNC_RL_PARENT_ATTEMPT_ID": (
                        str(resume_boundary["attempt_id"])
                        if resume_boundary is not None
                        else ""
                    ),
                    "DRIFT_ASYNC_RL_PARENT_COMMIT_ID": (
                        str(resume_boundary.get("scientific_commit_id") or "")
                        if resume_boundary is not None
                        else ""
                    ),
                    "DRIFT_ASYNC_RL_RESUME_BOUNDARY": (
                        str(
                            Path(logical_save_root)
                            / str(resume_boundary["scientific_commit_path"])
                        )
                        if resume_boundary is not None
                        and resume_boundary.get("scientific_commit_path")
                        else ""
                    ),
                }
            }
            # Credentials are inherited from the Ray daemons' ambient
            # environment, never serialized into Ray Job metadata. Reject the
            # generic recipe environment as an accidental reintroduction path.
            for sensitive_name in (
                "WANDB_API_KEY",
                "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN",
            ):
                runtime_env["env_vars"].pop(sensitive_name, None)

            mode = "async" if slime.async_mode else "sync"
            print(
                f"Training {app_name} — {slime.total_nodes} node(s) × {gpu_spec}  ({mode})"
            )
            print(slime.gpu_allocation.summary())
            print(f"Command: {cmd}, runtime_env: {redact_runtime_env(runtime_env)}")

            await _set_framework_status_async(SlimeStatus.ROLLOUT_INITIALIZING)
            if terminal_parent_complete:
                assert resume_boundary is not None
                print(
                    "Authenticated terminal boundary already covers the requested "
                    f"{slime.num_rollout} rollouts; finalizing its accepted parent "
                    "without replaying training."
                )
            else:
                ray_failure_stage = "ray_dashboard_setup"
                async with cluster.forward_dashboard() as tunnel:
                    print(f"Ray dashboard: {tunnel.url}")
                    ray_failure_stage = "ray_job_submission_or_streaming"
                    result = await cluster.submit_and_tail(cmd, runtime_env=runtime_env)
                    ray_failure_stage = "ray_job_terminal_result"
                    if not result.is_success:
                        attempt_error = (
                            result.message
                            or f"Ray job finished with status: {result.status}"
                        )
                        if result.diagnostics is not None:
                            ray_diagnostic_recorded = record_ray_failure_diagnostic(
                                run_record,
                                result.diagnostics,
                                attempt_id=attempt_id,
                                attempt_count=attempt_count,
                                ray_job_id=result.job_id,
                                ray_job_status=result.status,
                                failure_stage=ray_failure_stage,
                            )
                        primary_error = record_attempt_failure(
                            run_record,
                            attempt_error,
                            attempt_id=attempt_id,
                            attempt_count=attempt_count,
                        )
                        if primary_error != attempt_error:
                            print(
                                "Secondary retry failure (primary failure preserved): "
                                f"{attempt_error}",
                                flush=True,
                            )
                        raise RuntimeError(primary_error)
                    print(f"Ray job completed: {result.status}")
                    ray_failure_stage = "post_ray_job_finalization"

            if committed_attempt_mode:
                accepted_final_attempt_id = (
                    str(resume_boundary["attempt_id"])
                    if terminal_parent_complete and resume_boundary is not None
                    else attempt_id
                )
                accepted_lineage = write_accepted_lineage(
                    logical_save_root,
                    final_attempt_id=accepted_final_attempt_id,
                    final_rollout_id=slime.num_rollout - 1,
                )
                print(f"Accepted attempt lineage saved: {accepted_lineage}")

            result_checkpoint_dir = (
                str(attempt_root(logical_save_root, resume_boundary["attempt_id"]))
                if terminal_parent_complete and resume_boundary is not None
                else save_root
            )
            result = build_train_result(
                app_name=app_name,
                framework=Framework.SLIME,
                training_run_id=training_run_id,
                checkpoint_dir=result_checkpoint_dir,
                model=model,
                checkpoints_volume_name=checkpoints_volume_name,
                checkpoints_mount_path=checkpoints_mount_path,
                wandb_cfg=slime.wandb,
                wandb_entity=wandb_entity,
                wandb_run_id=wandb_run_id,
                group_id=group_id,
            )
            run_record.status = TrainingRunStatus.COMPLETED
            mark_training_attempt_finished(
                run_record,
                status=(
                    "finalized_from_terminal_parent"
                    if terminal_parent_complete
                    else "completed"
                ),
                ended_at=int(time.time()),
            )
            # Publish acceptance only after its checkpoint namespace and
            # accepted-lineage receipt are durable. The bound result is then
            # an immutable, idempotent re-entry marker for the logical run.
            await checkpoints_volume.commit.aio()
            if committed_attempt_mode:
                bind_accepted_train_result(
                    result,
                    run_contract_sha256=scientific_run_contract_sha256,
                    accepted_attempt_id=accepted_final_attempt_id,
                )
                await result.save(is_async=True, immutable=True)
            else:
                await result.save(is_async=True)
            print(f"TrainResult saved: {training_run_id}")
            return result._to_dict()
        except KeyboardInterrupt:
            mark_run_stopped(run_record)
            raise
        except BaseException as exc:
            if committed_attempt_mode and logical_save_root:
                try:
                    failure_boundary = load_latest_committed_boundary(
                        logical_save_root,
                        verify_hashes=False,
                        expected_run_contract_sha256=scientific_run_contract_sha256,
                    )
                    record_last_committed_boundary_snapshot(
                        run_record,
                        failure_boundary,
                        active_attempt_id=attempt_id,
                    )
                except Exception as boundary_exc:  # noqa: BLE001
                    print(
                        "Failed to read the last committed boundary while preserving "
                        f"the original launcher exception: "
                        f"{type(boundary_exc).__name__}: {boundary_exc}",
                        flush=True,
                    )
            try:
                if not ray_diagnostic_recorded:
                    capture_and_record_ray_failure_diagnostic(
                        run_record,
                        capture_ray_cluster_diagnostics,
                        attempt_id=attempt_id,
                        attempt_count=attempt_count,
                        ray_job_id=cluster.last_submitted_job_id,
                        ray_job_status=(
                            "CLUSTER_BOOTSTRAP_FAILED"
                            if not ray_started
                            else "LAUNCHER_EXCEPTION"
                        ),
                        failure_stage=ray_failure_stage,
                    )
            except Exception as diagnostic_exc:  # noqa: BLE001
                print(
                    "Failed to attach Ray diagnostics while preserving the "
                    f"original launcher exception: {type(diagnostic_exc).__name__}: "
                    f"{diagnostic_exc}",
                    flush=True,
                )
            primary_error = mark_run_failed(run_record, exc)
            try:
                # Publish the causal failure before unwinding. The later
                # terminal finalizer enriches this record, but a second failure
                # during cleanup cannot erase the first immutable event.
                await run_record.save(is_async=True, event_kind="failure")
            except Exception as failure_save_exc:  # noqa: BLE001
                print(
                    "Failed to journal the launcher failure while preserving "
                    f"the original exception: {type(failure_save_exc).__name__}: "
                    f"{failure_save_exc}",
                    flush=True,
                )
            current_error = f"{type(exc).__name__}: {exc}"
            if primary_error not in {str(exc).strip(), current_error}:
                print(
                    "Retry attempt failed after the causal failure; surfacing "
                    f"the preserved primary failure: {primary_error}",
                    flush=True,
                )
                raise RuntimeError(primary_error) from exc
            raise
        finally:
            try:
                # Diagnostics and the causal failure event are captured above.
                # Stop the head now so worker ranks can observe liveness loss
                # and return instead of idling until Modal tears down the app.
                cluster.stop_ray()
            except Exception as exc:  # noqa: BLE001 - never mask the training result
                print(
                    "Failed to stop Ray while preserving the training result: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            latest_run_record = await build_terminal_run_record(
                run_record, training_run_id
            )
            if os.environ.get("DRIFT_ASYNC_RL_D1_MATRIX") == "1":
                latest_run_record = _restore_d1a_terminal_binding(
                    run_record,
                    latest_run_record,
                )
            latest_attempt_id = str(
                (latest_run_record.metadata or {}).get("active_attempt_id") or ""
            )
            stale_finalizer = bool(
                attempt_id and latest_attempt_id and latest_attempt_id != attempt_id
            )

            step_times_read = False
            if not stale_finalizer:
                try:
                    latest_run_record.step_times = await write_step_times(
                        training_run_id,
                        attempt_id,
                        slime.num_rollout,
                    )
                    step_times_read = True
                except Exception as exc:
                    print(f"Failed to read step times: {exc}")

                terminal_d1a_success = (
                    os.environ.get("DRIFT_ASYNC_RL_D1_MATRIX") == "1"
                    and latest_run_record.status == TrainingRunStatus.COMPLETED
                )
                try:
                    if terminal_d1a_success:
                        await _persist_and_verify_d1a_terminal_success(
                            latest_run_record,
                            expected_attempt_id=attempt_id,
                        )
                    else:
                        await latest_run_record.save(is_async=True)
                except Exception as exc:
                    if terminal_d1a_success:
                        raise RuntimeError(
                            "D1a completed scientific work but terminal metadata "
                            "persistence/readback failed"
                        ) from exc
                    print(f"Failed to save run record: {exc}")
                else:
                    if step_times_read:
                        try:
                            await clear_step_times(
                                training_run_id,
                                attempt_id,
                                slime.num_rollout,
                            )
                        except Exception as exc:
                            print(f"Failed to clear step times: {exc}")

    for tag, fn in app.registered_functions.items():
        setattr(app, tag, fn)

    # ``image_env`` has already been baked into ``image``/``train_image`` above.
    # Keeping it on the recipe captured by these nested ``serialized=True``
    # functions needlessly duplicates every image environment value in Modal's
    # cloudpickle payload.  In particular, request-bearing experiments can
    # exceed Modal's 64 KiB serialized-function limit before a container starts.
    # Rebind the shared closure cell to a shallow copy so the caller's resolved
    # recipe and the exact image environment remain untouched.  Runtime code in
    # these functions consumes ``environment``; ``image_env`` is build-only.
    serialized_recipe = copy.copy(slime)
    object.__setattr__(serialized_recipe, "image_env", {})
    slime = serialized_recipe

    return app
