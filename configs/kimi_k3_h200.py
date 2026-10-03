"""Kimi-K3 LoRA 64k recipe on eight H200:8 nodes."""

from pathlib import Path

from modal_dojo import Kimi_K3_LoRA_Recipe
from modal_dojo.common.patches import encode_patch

from configs.kimi_k3_long_context import context_settings


def build_recipe(*, context_length=65536, concurrency_per_engine=1, **overrides):
    if context_length != 65536 or concurrency_per_engine != 1:
        raise ValueError(
            "The H200 candidate is scoped to 64k and one request per engine"
        )
    settings = context_settings(context_length, concurrency_per_engine)
    settings["gpu_type"] = "H200"
    patches = (
        Path(__file__).resolve().parents[1]
        / "modal_dojo/frameworks/miles/modal_helpers/patches"
    )
    compact_mxfp4 = (
        f"echo {encode_patch('patch_k3_marlin_padding', patches)} | base64 -d | python3"
    )
    cpu_lora_staging = f"echo {encode_patch('patch_k3_lora_cpu_staging', patches)} | base64 -d | python3"
    lora_health = (
        f"echo {encode_patch('patch_k3_lora_health', patches)} | base64 -d | python3"
    )
    cpu_checkpoint_merge = f"echo {encode_patch('patch_k3_checkpoint_cpu_merge', patches)} | base64 -d | python3"
    fused_activation = f"echo {encode_patch('patch_k3_fused_activation', patches)} | base64 -d | python3"
    settings.update(
        # Both compact inference and trainer backups total ~1.59 TiB/node.
        # H200 AWS hosts have 2 TiB; leave room for the host and runtime.
        memory=(1792 * 1024, 1920 * 1024),
        recompute_num_layers=3,
        optimizer_offload_fraction=1.0,
        sglang_mem_fraction_static=0.95,
        sglang_decode_attention_backend="flashinfer",
        image_run_commands=[
            *(overrides.pop("image_run_commands", None) or []),
            compact_mxfp4,
            # A whole unsharded adapter cannot fit alongside the frozen
            # serving weights. Stage independent IPC copies on the host;
            # SGLang validates and TP-slices them into its existing pool.
            cpu_lora_staging,
            lora_health,
            cpu_checkpoint_merge,
            fused_activation,
        ],
    )
    settings["extra_config"].update(overrides.pop("extra_config", None) or {})
    settings.update(overrides)
    recipe = Kimi_K3_LoRA_Recipe(**settings)
    if recipe.gpu_type != "H200" or recipe.context_parallel_size != 2:
        raise ValueError("The H200 recipe requires H200 GPUs and CP2")
    return recipe
