from __future__ import annotations

import json
from typing import Any, Literal

from modal_dojo.common.errors import DojoConfigError

Framework = Literal["slime", "miles"]


def _effective(config: Any, field_name: str, default: Any = None) -> Any:
    hatch = getattr(config, "_escape_hatch_values", None)
    values = hatch() if callable(hatch) else {}
    if field_name in values:
        return values[field_name]
    return getattr(config, field_name, default)


def _torch_memory_saver_offloads(
    config: Any, framework: Framework
) -> tuple[bool, bool]:
    # miles applies extra_config before resolving offload; slime applies it after.
    get = _effective if framework == "miles" else getattr
    sft = getattr(config, "loss_type", None) == "sft_loss"
    offload_train = get(config, "offload_train", None)
    offload_rollout = get(config, "offload_rollout", None)
    if getattr(config, "no_offload_train", False):
        offload_train = False
    if getattr(config, "no_offload_rollout", False):
        offload_rollout = False
    if get(config, "offload", None):
        offload_train = offload_rollout = True
    debug_rollout_only = get(config, "debug_rollout_only", None)
    if debug_rollout_only:
        offload_train = offload_rollout = False
    elif not sft and get(config, "colocate", False):
        if framework == "slime" and get(config, "release_train", None):
            offload_train, offload_rollout = False, True
        if offload_train is None:
            offload_train = True
        if offload_rollout is None:
            offload_rollout = True
    if not sft and get(config, "advantage_estimator", None) == "ppo":
        if framework == "slime":
            offload_train = True
        elif not debug_rollout_only and offload_train is None:
            offload_train = True
    if framework == "slime":
        offload_train = _effective(config, "offload_train", offload_train)
        offload_rollout = _effective(config, "offload_rollout", offload_rollout)
    return bool(offload_train), bool(offload_rollout)


def validate_expandable_segments_env(config: Any, framework: Framework) -> None:
    offload_train, offload_rollout = _torch_memory_saver_offloads(config, framework)
    train_env = _env_dict(_effective(config, "train_env_vars")) or {}
    for field_name in ("environment", "image_env", "train_env_vars"):
        if field_name == "train_env_vars":
            env = train_env
            reaches_train, reaches_rollout = True, False
        else:
            env = _env_dict(getattr(config, field_name, None)) or {}
            reaches_train = "PYTORCH_CUDA_ALLOC_CONF" not in train_env
            reaches_rollout = True
        if not _alloc_conf_enables_expandable_segments(
            env.get("PYTORCH_CUDA_ALLOC_CONF")
        ):
            continue
        if offload_train and reaches_train:
            offload = "offload_train"
        elif offload_rollout and reaches_rollout:
            offload = "offload_rollout"
        else:
            continue
        raise DojoConfigError(
            f"{field_name} sets expandable_segments with {offload} on, which "
            "TorchMemorySaver does not support; remove expandable_segments "
            f"from PYTORCH_CUDA_ALLOC_CONF or disable {offload}."
        )


def validate_loss_and_sampling_args(config: Any, framework: Framework) -> None:
    if getattr(config, "loss_type", None) == "sft_loss":
        return
    kl_coef = _effective(config, "kl_coef") or 0
    if kl_coef and _effective(config, "kl_loss_coef"):
        raise DojoConfigError(
            "kl_coef and kl_loss_coef are both nonzero; only one can be set."
        )
    if (
        framework == "miles"
        and kl_coef
        and _effective(config, "advantage_estimator") == "ppo"
    ):
        raise DojoConfigError(
            "advantage_estimator='ppo' does not support kl_coef; set kl_coef=0 "
            "and use use_kl_loss with kl_loss_coef instead."
        )
    over_sampling = _effective(config, "over_sampling_batch_size")
    rollout_batch_size = _effective(config, "rollout_batch_size")
    if (
        type(over_sampling) is int
        and type(rollout_batch_size) is int
        and over_sampling < rollout_batch_size
    ):
        raise DojoConfigError(
            f"over_sampling_batch_size={over_sampling} must be at least "
            f"rollout_batch_size={rollout_batch_size}."
        )


def _env_dict(env: Any) -> dict | None:
    if isinstance(env, str):
        try:
            env = json.loads(env)
        except json.JSONDecodeError:
            return None
    return env if isinstance(env, dict) else None


def _alloc_conf_enables_expandable_segments(alloc_conf: Any) -> bool:
    if not isinstance(alloc_conf, str):
        return False
    for item in alloc_conf.split(","):
        key, _, value = item.partition(":")
        if key.strip() == "expandable_segments":
            return value.strip().lower() in ("true", "1", "yes", "on")
    return False
