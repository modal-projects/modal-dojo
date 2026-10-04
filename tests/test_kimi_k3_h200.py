import pytest

from modal_dojo import Kimi_K3_LoRA_Recipe
from configs.kimi_k3_h200 import build_recipe


def test_h200_profile_preserves_base_patches_and_uses_memory_saving_settings():
    base = Kimi_K3_LoRA_Recipe()
    recipe = build_recipe(image_run_commands=["echo custom"])
    assert (
        recipe.image_run_commands[: len(base.image_run_commands)]
        == base.image_run_commands
    )
    assert "echo custom" in recipe.image_run_commands
    assert len(recipe.image_run_commands) == len(base.image_run_commands) + 8
    assert recipe.gpu_type == "H200"
    assert recipe.memory == (1792 * 1024, 1920 * 1024)
    assert recipe.actor_num_nodes == recipe.actor_num_gpus_per_node == 8
    assert recipe.rollout_num_gpus_per_engine == 16
    assert recipe.tensor_model_parallel_size == 2
    assert recipe.context_parallel_size == 4
    assert recipe.pipeline_model_parallel_size == 8
    assert recipe.max_tokens_per_gpu == 16384
    assert recipe.extra_config["log_probs_max_tokens_per_gpu"] == 16384
    assert recipe.extra_config["fp8"] == "hybrid"
    assert recipe.extra_config["fp8_recipe"] == "delayed"
    assert recipe.extra_config["fp8_param_gather"]
    assert recipe.recompute_granularity == "full"
    assert recipe.recompute_method == "uniform"
    assert recipe.recompute_num_layers == 1
    assert recipe.optimizer_offload_fraction == 1.0
    assert recipe.sglang_mem_fraction_static == 0.95
    assert recipe.sglang_decode_attention_backend == "flashinfer"
    assert recipe.colocate_memory_peak_device == "cpu"
    assert recipe.experts_shared_outer_loras
    assert recipe.ref_load == base.ref_load
    assert recipe.docker_image == base.docker_image


@pytest.mark.parametrize(
    "overrides", [{"context_length": 131072}, {"concurrency_per_engine": 2}]
)
def test_h200_profile_requires_new_capacity_check_for_larger_shapes(overrides):
    with pytest.raises(ValueError, match="H200 candidate"):
        build_recipe(**overrides)


def test_rejects_other_hardware():
    with pytest.raises(ValueError, match="H200 recipe"):
        build_recipe(gpu_type="B300")
