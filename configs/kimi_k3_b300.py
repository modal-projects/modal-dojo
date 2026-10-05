"""Kimi-K3 LoRA long-context recipe on eight B300:8 nodes."""

from modal_dojo import Kimi_K3_LoRA_Recipe

from configs.kimi_k3_long_context import context_settings


def build_recipe(*, context_length=65536, concurrency_per_engine=1, **overrides):
    if context_length not in (65536, 131072):
        raise ValueError("Choose a target shape: 65536 or 131072 tokens")
    if concurrency_per_engine not in (1, 2, 4, 8):
        raise ValueError("concurrency_per_engine must be one of 1, 2, 4, 8")
    settings = context_settings(context_length, concurrency_per_engine)
    settings.update(gpu_type="B300", memory=(2560 * 1024, 3072 * 1024))
    settings["extra_config"].update(overrides.pop("extra_config", None) or {})
    settings.update(overrides)
    recipe = Kimi_K3_LoRA_Recipe(**settings)
    if recipe.gpu_type != "B300" or recipe.context_parallel_size != 2:
        raise ValueError("The B300 recipe requires B300 GPUs and CP2")
    return recipe
