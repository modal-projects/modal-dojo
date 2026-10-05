"""Kimi-K3 LoRA long-context recipe on eight B300:8 nodes."""

from modal_dojo import Kimi_K3_LoRA_Recipe
from modal_dojo.common.patches import encode_patch
from modal_dojo.train_recipes.miles_recipe.kimi_k3 import _PATCH_DIR

from configs.kimi_k3_long_context import context_settings


def retained_backup_image_commands():
    """Build the exact allocator revision in the pinned CUDA 13 image."""
    return [
        "git clone https://github.com/fzyzcjy/torch_memory_saver.git /tmp/dojo-tms "
        "&& git -C /tmp/dojo-tms checkout b5588e83de86412a48689a6583a4b567e75f7acc",
        f"echo {encode_patch('patch_tms_retain_backup', _PATCH_DIR)} | base64 -d | python3",
        "TMS_CUDA_MAJOR=13 uv pip install --python /opt/sglang/bin/python "
        "--no-deps --no-build-isolation --reinstall /tmp/dojo-tms",
    ]


def build_recipe(*, context_length=65536, concurrency_per_engine=2, **overrides):
    if context_length not in (65536, 131072):
        raise ValueError("Choose a target shape: 65536 or 131072 tokens")
    if concurrency_per_engine not in (1, 2, 4, 8):
        raise ValueError("concurrency_per_engine must be one of 1, 2, 4, 8")
    settings = context_settings(context_length, concurrency_per_engine)
    settings.update(gpu_type="B300", memory=(2560 * 1024, 3072 * 1024))
    settings["extra_config"].update(overrides.pop("extra_config", None) or {})
    # The 3 TiB host limit accommodates retained inference backups. Keep this
    # opt-in to the long-context B300 configuration, not every K3/H200 recipe.
    environment = dict(Kimi_K3_LoRA_Recipe().environment)
    environment["DOJO_TMS_RETAIN_BACKUP_TAG"] = "weights"
    environment["DOJO_LOCAL_KERNEL_CACHE"] = "/tmp/dojo-kernel-cache/dev-202609251434"
    environment.update(overrides.pop("environment", None) or {})
    settings["environment"] = environment
    settings["image_run_commands"] = [
        f"echo {encode_patch('patch_k3_marlin_padding', _PATCH_DIR)} | base64 -d | python3",
        f"echo {encode_patch('patch_k3_lora_health', _PATCH_DIR)} | base64 -d | python3",
        *retained_backup_image_commands(),
        *(overrides.pop("image_run_commands", None) or []),
    ]
    settings.update(overrides)
    recipe = Kimi_K3_LoRA_Recipe(**settings)
    if recipe.gpu_type != "B300" or recipe.context_parallel_size != 2:
        raise ValueError("The B300 recipe requires B300 GPUs and CP2")
    return recipe
