import pytest

from configs.kimi_k3_b300 import build_recipe
from configs.kimi_k3_long_context import generate_with_context_limit


@pytest.mark.parametrize("context", [65536, 131072])
def test_context_config_keeps_base_identity_and_patches(context):
    from modal_dojo import Kimi_K3_LoRA_Recipe

    base = Kimi_K3_LoRA_Recipe()
    recipe = build_recipe(context_length=context)
    assert type(recipe) is type(base)
    assert recipe.image_run_commands == base.image_run_commands
    assert recipe.environment == base.environment
    assert recipe.custom_generate_function is generate_with_context_limit
    assert recipe.ref_load == base.ref_load
    assert recipe.max_tokens_per_gpu * recipe.context_parallel_size == context
    assert recipe.extra_config["sglang_context_length"] == context
    assert (
        recipe.extra_config["seq_length"]
        == recipe.extra_config["max_position_embeddings"]
        == context
    )
    assert recipe.extra_config["rollout_max_prompt_len"] is None
    assert recipe.rollout_max_response_len == recipe.eval_max_response_len == context
    assert recipe.sglang_max_total_tokens > context
    assert recipe.gpu_allocation == base.gpu_allocation
    assert recipe.colocate_memory_peak_device == "cpu"
    assert recipe.memory == (2560 * 1024, 3072 * 1024)
    engines = (
        recipe.actor_num_nodes
        * recipe.actor_num_gpus_per_node
        // recipe.rollout_num_gpus_per_engine
    )
    assert recipe.miles_router_max_connections == base.miles_router_max_connections
    assert recipe.miles_router_max_connections >= engines * (
        recipe.sglang_server_concurrency + 1
    )
    assert "LD_LIBRARY_PATH" not in recipe.environment


def test_rejects_other_hardware():
    with pytest.raises(ValueError, match="B300 recipe"):
        build_recipe(gpu_type="H200")
