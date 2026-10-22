"""A workload-independent agent recipe with explicit optional disk reuse."""

import base64

import pytest

from modal_dojo import (
    DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe,
    DeepSeek_V4_1_Flash_LoRA_Recipe,
)


def test_agent_recipe_preserves_base_hardware_and_training_configuration():
    base = DeepSeek_V4_1_Flash_LoRA_Recipe()
    recipe = DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe()
    for name in (
        "docker_image",
        "miles_git_ref",
        "sglang_git_ref",
        "actor_num_nodes",
        "actor_num_gpus_per_node",
        "tensor_model_parallel_size",
        "expert_model_parallel_size",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "lr",
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "use_rollout_routing_replay",
    ):
        assert getattr(recipe, name) == getattr(base, name)
    assert recipe.rollout_max_prompt_len + recipe.rollout_max_response_len == 32768
    assert recipe.rollout_max_response_len == 2048
    assert base.rollout_max_response_len == 28672
    assert (
        not recipe.no_save_optim and not recipe.no_load_optim and not recipe.no_load_rng
    )
    assert recipe.extra_config["sglang_router_policy"] == "manual"
    assert recipe.extra_config["router_assignment_mode"] == "min_load"
    assert base.extra_config["sglang_router_policy"] == "round_robin"
    assert recipe.environment["DOJO_UPDATE_DIAGNOSTICS"] == "1"
    assert not any(key.startswith("TAU_") for key in recipe.environment)
    assert (
        recipe.rollout_function is None
        and recipe.custom_reward_post_process_function is None
    )


def test_image_patches_have_no_harness_dependency_and_optional_reuse_is_off():
    recipe = DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe()
    sources = [
        base64.b64decode(command.split()[1]).decode()
        for command in recipe.image_run_commands
    ]
    assert any("patch_token_normalizer" in source for source in sources)
    assert any("capture_resume_state" in source for source in sources)
    assert any("lora_diagnostics" in source for source in sources)
    assert all(
        "import tau_quest" not in source and "from tau_quest" not in source
        for source in sources
    )
    assert not any("TMS_ARCHIVE_SHA256" in source for source in sources)


def test_reuse_builds_serving_extension_and_is_not_a_miles_cli_flag():
    recipe = DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe(reuse_frozen_base_backup=True)
    assert recipe.environment["DOJO_REUSE_FROZEN_BASE_BACKUP"] == "1"
    assert "reuse_frozen_base_backup" in recipe._SKIP_FIELDS
    assert any(
        "TMS_ARCHIVE_SHA256" in base64.b64decode(command.split()[1]).decode()
        for command in recipe.image_run_commands
    )
    with pytest.raises(ValueError, match="requires native CUDA LoRA"):
        DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe(
            reuse_frozen_base_backup=True, extra_config=None
        )
    with pytest.raises(ValueError, match="Enable snapshot reuse"):
        DeepSeek_V4_1_Flash_LoRA_Agentic_Recipe(
            environment={"DOJO_REUSE_FROZEN_BASE_BACKUP": "1"}
        )
