"""Translate a :class:`SpindleRecipe` into Spindle deployment settings.

Pure Python: no ``spindle`` import, so the mapping is testable locally. The
result is fed to ``spindle.configuration.BaseConfig(**values)`` inside the
deployment container.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from modal_dojo.train_recipes.spindle_recipe.recipe import SpindleRecipe

MODEL_TYPES = {"Qwen/Qwen3.8-27B": "qwen3.8-27B"}

# Modal GPU names are case-insensitive but Spindle compares them literally.
_GPU_NAMES = {"H100": "H100", "H200": "H200", "B200": "B200", "A100": "A100-80GB"}


def apply_dotted_overrides(values: dict[str, Any], overrides: dict[str, Any]) -> None:
    for path, value in overrides.items():
        parts = path.split(".")
        target = values
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = deepcopy(value)


def _gpu_name(gpu_type: str) -> str:
    base = gpu_type.split(":")[0].strip().rstrip("!+")
    return _GPU_NAMES.get(base.upper(), base)


def _memory_mib(memory: int | tuple[int, int] | None, default: int) -> int:
    if memory is None:
        return default
    return int(memory[0] if isinstance(memory, tuple) else memory)


def _cpu(cpu: float | tuple[float, float] | None, default: int) -> int:
    if cpu is None:
        return default
    return int(cpu[0] if isinstance(cpu, tuple) else cpu)


def miles_cfg(recipe: SpindleRecipe, model_type: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model_type": model_type,
        "tensor_model_parallel_size": recipe.tensor_model_parallel_size,
        "max_tokens_per_gpu": recipe.max_tokens_per_gpu,
        "max_lora_slots": 1,
        "max_lora_rank": recipe.lora_rank,
        "default_lora_alpha": recipe.lora_alpha or recipe.lora_rank,
    }
    if recipe.target_modules:
        cfg["target_modules"] = [
            name.strip() for name in recipe.target_modules.split(",") if name.strip()
        ]
    if recipe.pipeline_model_parallel_size > 1:
        cfg["pipeline_model_parallel_size"] = recipe.pipeline_model_parallel_size
    if recipe.context_parallel_size > 1:
        cfg["context_parallel_size"] = recipe.context_parallel_size
    if recipe.expert_model_parallel_size > 1:
        cfg["expert_model_parallel_size"] = recipe.expert_model_parallel_size
    cli_options: dict[str, Any] = {}
    for name in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
        value = getattr(recipe, name)
        if value is not None:
            cli_options[name] = value
    if recipe.sequence_parallel:
        cli_options["sequence_parallel"] = True
    if recipe.attention_backend:
        cli_options["attention_backend"] = recipe.attention_backend
    if recipe.lora_dropout:
        cli_options["lora_dropout"] = recipe.lora_dropout
    if cli_options:
        cfg["cli_options"] = cli_options
    return cfg


def sglang_cfg(recipe: SpindleRecipe) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "tp_size": recipe.rollout_num_gpus_per_engine,
        "mem_fraction_static": recipe.sglang_mem_fraction_static,
    }
    if recipe.sglang_max_running_requests is not None:
        cfg["max_running_requests"] = recipe.sglang_max_running_requests
    if recipe.sglang_enable_dp_attention:
        cfg["enable_dp_attention"] = True
    if recipe.sglang_dp_size:
        cfg["dp_size"] = recipe.sglang_dp_size
    if recipe.sglang_ep_size:
        cfg["ep_size"] = recipe.sglang_ep_size
    if recipe.sglang_attention_backend:
        cfg["attention_backend"] = recipe.sglang_attention_backend
    if recipe.sglang_disable_cuda_graph:
        cfg["disable_cuda_graph"] = True
    if recipe.sglang_disable_radix_cache:
        cfg["disable_radix_cache"] = True
    if recipe.sglang_disable_custom_all_reduce:
        cfg["disable_custom_all_reduce"] = True
    if recipe.sglang_lora_backend:
        cfg["lora_backend"] = recipe.sglang_lora_backend
    return cfg


def spindle_config_values(
    recipe: SpindleRecipe,
    *,
    model_name: str,
    training_run_id: str = "",
    modal_environment: str | None = None,
) -> dict[str, Any]:
    """Keyword arguments for ``spindle.configuration.BaseConfig``."""
    model_type = MODEL_TYPES.get(model_name)
    if model_type is None:
        raise ValueError(f"No Spindle model_type registered for {model_name!r}")
    gpu = _gpu_name(recipe.gpu_type)
    environment = recipe.spindle_environment or modal_environment
    values: dict[str, Any] = {
        "name": recipe.spindle_deployment_name(training_run_id),
        "model": model_name,
        "max_context_length": recipe.max_context_length,
        "parameterization": "lora",
        "backend": "miles",
        "trainer_gpu": gpu,
        "trainer_gpus_per_node": recipe.actor_num_gpus_per_node,
        "trainer_nodes": recipe.actor_num_nodes,
        "trainer_cpu": _cpu(recipe.cpu, 8),
        "trainer_memory_mib": _memory_mib(recipe.memory, 32768),
        "trainer_max_instances": 1,
        "trainer_max_clients_per_instance": 1,
        "trainer_env": dict(recipe.environment),
        "inference_gpu": gpu,
        "inference_gpus_per_node": recipe.rollout_num_gpus_per_engine,
        "inference_min_replicas": 0,
        "inference_max_replicas": recipe.inference_replicas,
        "inference_target_concurrency": max(
            1, (recipe.sglang_max_running_requests or 32) // 2
        ),
        "miles_cfg": miles_cfg(recipe, model_type),
        "sglang_cfg": sglang_cfg(recipe),
        "platform": {
            "frontend": recipe.spindle_frontend_name(training_run_id),
            "modal": {"environment": environment, "region": recipe.region},
            "secrets": {
                "api": recipe.spindle_api_secret,
                "sampler_proxy": recipe.spindle_proxy_secret,
                "huggingface": "huggingface-secret",
            },
            "storage": {
                "assets": "spindle-model-assets",
                "checkpoints": "spindle-checkpoints",
                "bulletin": "spindle-snapshot-bulletin",
            },
        },
    }
    apply_dotted_overrides(values["platform"], recipe.spindle_platform_overrides)
    apply_dotted_overrides(values, recipe.spindle_config_overrides)
    return values
