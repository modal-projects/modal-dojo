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
    fused_lora = (
        f"echo {encode_patch('patch_k3_fused_lora', patches)} | base64 -d | python3"
    )
    fp8_checkpoint = (
        f"echo {encode_patch('patch_k3_fp8_checkpoint', patches)} | base64 -d | python3"
    )
    settings.update(
        memory=(1792 * 1024, 1920 * 1024),
        tensor_model_parallel_size=2,
        context_parallel_size=4,
        max_tokens_per_gpu=context_length // 4,
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        global_batch_size=4,
        recompute_num_layers=1,
        optimizer_offload_fraction=1.0,
        sglang_mem_fraction_static=0.95,
        sglang_decode_attention_backend="flashinfer",
        image_run_commands=[
            *(overrides.pop("image_run_commands", None) or []),
            compact_mxfp4,
            cpu_lora_staging,
            lora_health,
            cpu_checkpoint_merge,
            fused_activation,
            fused_lora,
            fp8_checkpoint,
        ],
    )
    settings["extra_config"]["log_probs_max_tokens_per_gpu"] = context_length // 4

    settings["extra_config"].update(
        fp8="hybrid", fp8_recipe="delayed", fp8_param_gather=True
    )
    settings["extra_config"].update(overrides.pop("extra_config", None) or {})
    settings.update(overrides)
    recipe = Kimi_K3_LoRA_Recipe(**settings)

    recipe.environment = {
        **recipe.environment,
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8",
        **(overrides.get("environment") or {}),
    }
    if (
        recipe.gpu_type != "H200"
        or recipe.context_parallel_size != 4
        or recipe.tensor_model_parallel_size != 2
    ):
        raise ValueError("The H200 recipe requires H200 GPUs and TP2/CP4")
    return recipe
