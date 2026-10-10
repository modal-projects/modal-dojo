from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from math import prod
from typing import Any

from modal_dojo.common.errors import DojoConfigError, GpuAllocationError

_MAX_GPUS_PER_CONTAINER = {"A10": 4}


@dataclass(frozen=True)
class GpuAllocation:
    actor_gpus: int
    critic_gpus: int
    rollout_gpus: int
    total_gpus: int
    total_nodes: int
    gpus_per_node: int
    rollout_num_gpus_per_engine: int
    rollout_engines: int
    colocate: bool

    def summary(self) -> str:
        parts = [
            f"actor={self.actor_gpus}",
            f"rollout={self.rollout_gpus}",
            f"critic={self.critic_gpus}",
            f"total={self.total_gpus}",
            f"nodes={self.total_nodes}",
        ]
        if self.rollout_gpus:
            parts.append(
                f"rollout_engines={self.rollout_engines}x{self.rollout_num_gpus_per_engine}"
            )
        return "GPU allocation: " + ", ".join(parts)


def _normalize_gpu_type(gpu_type: str | None) -> str:
    return (gpu_type or "").split(":")[0].strip().rstrip("!+").upper()


# per-device total as torch reports it (total_memory / 2**30)
_GPU_MEMORY_GIB = {
    "H100": 79.2,
    "H200": 140.4,
    "B200": 178.4,
    "B300": 268.0,
    "A100": 39.4,
    "A100-40GB": 39.4,
    "A100-80GB": 79.2,
    "L40S": 44.4,
    "A10": 22.1,
}


def gpu_memory_gib(gpu_type: str | None) -> float | None:
    return _GPU_MEMORY_GIB.get(_normalize_gpu_type(gpu_type))


def _max_gpus_per_container(gpu_type: str | None) -> int:
    return _MAX_GPUS_PER_CONTAINER.get(_normalize_gpu_type(gpu_type), 8)


def resolve_gpu_allocation(config: Any, *, warn: bool = True) -> GpuAllocation:
    gpus_per_node = _positive_int_field(config, "actor_num_gpus_per_node")
    actor_nodes = _positive_int_field(config, "actor_num_nodes")
    actor_gpus = actor_nodes * gpus_per_node
    rollout_num_gpus_per_engine = _positive_int_field(
        config, "rollout_num_gpus_per_engine"
    )

    colocate = bool(getattr(config, "colocate", False))
    critic_gpus = _critic_gpus(config, actor_nodes, gpus_per_node)
    rollout_gpus = _rollout_gpus(
        config,
        actor_gpus=actor_gpus,
        critic_gpus=critic_gpus,
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
        colocate=colocate,
        warn=warn,
    )

    total_gpus = actor_gpus + critic_gpus + rollout_gpus
    max_gpus = _max_gpus_per_container(getattr(config, "gpu_type", None))
    if actor_nodes == 1 and total_gpus <= max_gpus:
        gpus_per_node = total_gpus
        total_nodes = 1
    elif total_gpus % gpus_per_node != 0:
        raise GpuAllocationError(
            f"total_gpus={total_gpus} is not a multiple of gpus_per_node={gpus_per_node}. "
            "Adjust actor_num_nodes, rollout_num_gpus, or actor_num_gpus_per_node."
        )
    else:
        total_nodes = total_gpus // gpus_per_node
    rollout_engines = rollout_gpus // rollout_num_gpus_per_engine if rollout_gpus else 0

    return GpuAllocation(
        actor_gpus=actor_gpus,
        critic_gpus=critic_gpus,
        rollout_gpus=rollout_gpus,
        total_gpus=total_gpus,
        total_nodes=total_nodes,
        gpus_per_node=gpus_per_node,
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
        rollout_engines=rollout_engines,
        colocate=colocate,
    )


def validate_multi_node_gpu_count(
    allocation: GpuAllocation, gpu_type: str | None = None
) -> None:
    max_gpus = _max_gpus_per_container(gpu_type)
    if allocation.gpus_per_node > max_gpus:
        raise GpuAllocationError(
            f"gpus_per_node={allocation.gpus_per_node} exceeds the "
            f"{max_gpus} GPU container limit."
        )
    if allocation.total_nodes <= 1:
        return
    if allocation.gpus_per_node == max_gpus:
        return
    raise GpuAllocationError(
        f"Multi-node functions must use {max_gpus} GPUs per node; "
        f"requested {allocation.gpus_per_node}. Use actor_num_nodes=1 to pack "
        f"one node, or actor_num_gpus_per_node={max_gpus}."
    )


def validate_megatron_actor_parallelism(config: Any) -> None:
    actor_world_size = _positive_int_field(
        config, "actor_num_nodes"
    ) * _positive_int_field(config, "actor_num_gpus_per_node")

    model_parallel = _parallelism_fields(
        config,
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    )
    _require_actor_world_size_divisible(
        actor_world_size,
        model_parallel,
        label="tensor_model_pipeline_context_parallel",
        hint="actor_num_nodes, actor_num_gpus_per_node, TP, PP, or CP",
    )

    expert_parallel = _parallelism_fields(
        config,
        "expert_tensor_parallel_size",
        "expert_model_parallel_size",
        "pipeline_model_parallel_size",
    )
    _require_actor_world_size_divisible(
        actor_world_size,
        expert_parallel,
        label="expert_tensor_model_pipeline_parallel",
        hint="actor_num_nodes, actor_num_gpus_per_node, ETP, EP, or PP",
    )


def validate_microbatch_schedule(config: Any) -> None:
    actor_world = _positive_int_field(config, "actor_num_nodes") * _positive_int_field(
        config, "actor_num_gpus_per_node"
    )
    model_parallel = prod(
        _positive_int_knob(config, field_name, default=1)
        for field_name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        )
    )
    if actor_world % model_parallel:
        return
    dp_size = actor_world // model_parallel

    micro_batch_size = _positive_int_knob(config, "micro_batch_size", default=1)

    eval_global_batch_size = _eval_knob_value(config, "eval_global_batch_size")
    if eval_global_batch_size is not None:
        if type(eval_global_batch_size) is not int or eval_global_batch_size <= 0:
            raise DojoConfigError(
                f"eval_global_batch_size must be a positive integer, "
                f"got {eval_global_batch_size!r}"
            )
        eval_micro_batch_size = _eval_knob_value(config, "eval_micro_batch_size")
        if eval_micro_batch_size is None:
            eval_micro_batch_size = micro_batch_size
        elif type(eval_micro_batch_size) is not int or eval_micro_batch_size <= 0:
            raise DojoConfigError(
                f"eval_micro_batch_size must be a positive integer, "
                f"got {eval_micro_batch_size!r}"
            )
        if eval_global_batch_size % (eval_micro_batch_size * dp_size):
            raise DojoConfigError(
                f"eval_global_batch_size={eval_global_batch_size} must be "
                f"divisible by eval_micro_batch_size={eval_micro_batch_size} * "
                f"data_parallel_size={dp_size} "
                f"(actor world_size={actor_world} / TP*PP*CP={model_parallel})."
            )

    if _knob_value(config, "use_dynamic_batch_size", True):
        return

    if _knob_value(config, "loss_type", "policy_loss") == "sft_loss":
        hatch = getattr(config, "_escape_hatch_values", None)
        hatch_values = hatch() if callable(hatch) else {}
        resolved = next(
            (
                value
                for value in (
                    hatch_values.get("global_batch_size"),
                    hatch_values.get("rollout_batch_size"),
                    getattr(config, "global_batch_size", None),
                    getattr(config, "rollout_batch_size", None),
                )
                if value is not None
            ),
            None,
        )
        if resolved is None:
            resolved = _positive_int_knob(config, "global_batch_size")
        if type(resolved) is not int or resolved <= 0:
            raise DojoConfigError(
                f"global_batch_size must be a positive integer, got {resolved!r}"
            )
        global_batch_size = resolved
    else:
        global_batch_size = _positive_int_knob(config, "global_batch_size")

    micro_batches, remainder = divmod(global_batch_size, micro_batch_size)
    if remainder:
        raise DojoConfigError(
            f"static batch path: global_batch_size={global_batch_size} "
            f"is not a multiple of micro_batch_size={micro_batch_size}."
        )
    if micro_batches % dp_size:
        raise DojoConfigError(
            f"static batch path: global_batch_size={global_batch_size} "
            f"packs into {micro_batches} micro-batches of size {micro_batch_size}, "
            f"which is not a multiple of dp_size={dp_size} "
            f"(actor world_size={actor_world} / TP*PP*CP={model_parallel})."
        )

    if _knob_value(config, "loss_type", "policy_loss") != "sft_loss":
        rollout_samples = _positive_int_knob(
            config, "rollout_batch_size", default=global_batch_size
        ) * _positive_int_knob(config, "n_samples_per_prompt", default=1)
        if rollout_samples % global_batch_size:
            raise DojoConfigError(
                f"static batch path: rollout_batch_size x n_samples_per_prompt "
                f"yields {rollout_samples} samples per rollout, which is not a "
                f"multiple of global_batch_size={global_batch_size}."
            )


def _eval_knob_value(config: Any, field_name: str) -> Any:
    hatch = getattr(config, "_escape_hatch_values", None)
    if callable(hatch):
        value = hatch().get(field_name)
        if value is not None:
            return value
    eval_config = getattr(config, "eval_config", None)
    if isinstance(eval_config, dict):
        return eval_config.get(field_name)
    if isinstance(eval_config, str):
        import yaml

        try:
            if os.path.exists(eval_config):
                with open(eval_config) as f:
                    loaded = yaml.safe_load(f)
            else:
                loaded = yaml.safe_load(eval_config)
        except (OSError, yaml.YAMLError):
            return None
        if isinstance(loaded, dict):
            return loaded.get(field_name)
    return None


def validate_num_experts_divisible_by_expert_parallel_size(
    config: Any, model: Any
) -> None:
    architecture = getattr(model, "architecture", None)
    num_experts = getattr(architecture, "num_experts", 0) or 0
    if not num_experts:
        return

    if getattr(config, "expert_model_parallel_size", None) is None:
        # Unset EP falls back to the framework default of 1, which always divides.
        return

    expert_parallel_size = _positive_int_field(config, "expert_model_parallel_size")
    _require_divisible(
        num_experts,
        expert_parallel_size,
        value_name="num_experts",
        divisor_name="expert_model_parallel_size",
        hint="Adjust the model architecture or expert_model_parallel_size.",
    )


def _critic_gpus(config: Any, actor_nodes: int, gpus_per_node: int) -> int:
    if not bool(getattr(config, "use_critic", False)):
        return 0

    critic_nodes = _positive_int_field(config, "critic_num_nodes", default=actor_nodes)
    critic_gpus_per_node = _positive_int_field(
        config,
        "critic_num_gpus_per_node",
        default=gpus_per_node,
    )
    return critic_nodes * critic_gpus_per_node


def _rollout_gpus(
    config: Any,
    *,
    actor_gpus: int,
    critic_gpus: int,
    rollout_num_gpus_per_engine: int,
    colocate: bool,
    warn: bool,
) -> int:
    explicit_rollout_gpus = _optional_positive_int_field(config, "rollout_num_gpus")

    if getattr(config, "loss_type", None) == "sft_loss":
        if warn and explicit_rollout_gpus is not None:
            warnings.warn(
                "loss_type='sft_loss' does not start rollout engines; "
                f"rollout_num_gpus={explicit_rollout_gpus} is ignored.",
                stacklevel=2,
            )
        return 0

    if colocate:
        if warn and explicit_rollout_gpus not in (None, actor_gpus):
            warnings.warn(
                f"colocate=True uses actor GPUs for rollout; rollout_num_gpus={explicit_rollout_gpus} "
                f"does not change total allocation ({actor_gpus + critic_gpus} GPUs).",
                stacklevel=2,
            )
        return 0

    rollout_gpus = (
        actor_gpus if explicit_rollout_gpus is None else explicit_rollout_gpus
    )
    if warn and explicit_rollout_gpus is None:
        warnings.warn(
            f"colocate=False and rollout_num_gpus is unset; defaulting rollout allocation "
            f"to actor allocation ({rollout_gpus} GPUs).",
            stacklevel=2,
        )
    if rollout_gpus < rollout_num_gpus_per_engine:
        raise GpuAllocationError(
            f"rollout_num_gpus={rollout_gpus} is smaller than "
            f"rollout_num_gpus_per_engine={rollout_num_gpus_per_engine}"
        )
    _require_divisible(
        rollout_gpus,
        rollout_num_gpus_per_engine,
        value_name="rollout_num_gpus",
        divisor_name="rollout_num_gpus_per_engine",
    )
    if warn and rollout_gpus > actor_gpus * 2:
        warnings.warn(
            f"rollout allocation ({rollout_gpus} GPUs) is more than 2x actor allocation "
            f"({actor_gpus} GPUs).",
            stacklevel=2,
        )
    return rollout_gpus


def _parallelism_fields(config: Any, *field_names: str) -> dict[str, int]:
    return {
        field_name: _positive_int_field(config, field_name, default=1)
        for field_name in field_names
    }


def _require_actor_world_size_divisible(
    actor_world_size: int,
    parallelism: dict[str, int],
    *,
    label: str,
    hint: str,
) -> None:
    parallel_size = prod(parallelism.values())
    if actor_world_size % parallel_size == 0:
        return

    factors = ", ".join(f"{name}={value}" for name, value in parallelism.items())
    raise GpuAllocationError(
        f"actor world_size={actor_world_size} is not divisible by "
        f"{label} size={parallel_size} ({factors}). Adjust {hint}."
    )


def _require_divisible(
    value: int,
    divisor: int,
    *,
    value_name: str,
    divisor_name: str,
    hint: str | None = None,
) -> None:
    if value % divisor == 0:
        return

    message = f"{value_name}={value} is not divisible by {divisor_name}={divisor}"
    if hint:
        message += f". {hint}"
    raise GpuAllocationError(message)


def _knob_value(config: Any, field_name: str, default: Any = None) -> Any:
    hatch = getattr(config, "_escape_hatch_values", None)
    if callable(hatch):
        value = hatch().get(field_name)
        if value is not None:
            return value
    return getattr(config, field_name, default)


def _positive_int_knob(
    config: Any, field_name: str, *, default: int | None = None
) -> int:
    value = _knob_value(config, field_name)
    if value is None:
        value = default
    if value is None:
        raise DojoConfigError(f"{field_name} must be set")
    if type(value) is not int or value <= 0:
        raise DojoConfigError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _positive_int_field(
    config: Any, field_name: str, *, default: int | None = None
) -> int:
    value = getattr(config, field_name, default)
    if value is None:
        raise GpuAllocationError(f"{field_name} must be set")
    if type(value) is not int:
        raise GpuAllocationError(
            f"{field_name} must be an integer, got {type(value).__name__}"
        )
    if value <= 0:
        raise GpuAllocationError(f"{field_name} must be positive")
    return value


def _optional_positive_int_field(config: Any, field_name: str) -> int | None:
    value = getattr(config, field_name, None)
    if value is None:
        return None
    return _positive_int_field(config, field_name)
