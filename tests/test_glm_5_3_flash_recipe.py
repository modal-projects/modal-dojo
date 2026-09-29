"""GLM's custom model arguments must survive training and conversion assembly."""

from modal_training_gym import GLM_5_3_Flash, GLM_5_3_Flash_Recipe
from modal_training_gym.frameworks.miles.modal_helpers.utils import (
    build_train_cmd,
    get_checkpoint_conversion_policy,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def test_glm_custom_spec_and_pipeline_split_reach_training():
    model = GLM_5_3_Flash()
    recipe = MilesRecipe.get_base_recipe(model)
    assert isinstance(recipe, GLM_5_3_Flash_Recipe)
    assert model.architecture is None
    assert recipe.gpu_allocation.total_gpus == 32
    cmd = build_train_cmd(recipe, "/root/miles", model=model)
    assert "model_args_utils.py glm5.3-flash" in cmd
    assert "${MODEL_ARGS[@]}" in cmd
    assert "--model-name glm5_next" in cmd
    assert "--decoder-first-pipeline-num-layers 11" in cmd
    assert "--decoder-last-pipeline-num-layers 12" in cmd
    assert "--use-dynamic-batch-size" not in cmd
    assert "phase_reporting.before_log_prob_hook" in cmd
    assert "phase_reporting.before_train_step_hook" in cmd


def test_glm_conversion_shards_experts_without_training_pipeline_split():
    nodes, ranks, args = get_checkpoint_conversion_policy(
        GLM_5_3_Flash_Recipe(), GLM_5_3_Flash()
    )
    assert (nodes, ranks) == (1, 8)
    assert "--tensor-model-parallel-size 8" in args
    assert "--pipeline-model-parallel-size 1" in args
    assert "--expert-model-parallel-size 8" in args
    assert "--expert-tensor-parallel-size 1" in args
    assert not any("pipeline-num-layers" in arg for arg in args)


def test_glm_disk_reservation_survives_custom_function_kwargs():
    recipe = GLM_5_3_Flash_Recipe(train_function_kwargs={"timeout": 3600})
    assert recipe.train_function_kwargs == {
        "timeout": 3600,
        "ephemeral_disk": 2048 * 1024,
    }
    recipe = GLM_5_3_Flash_Recipe(train_function_kwargs={"ephemeral_disk": 2048 * 1024})
    assert recipe.train_function_kwargs["ephemeral_disk"] == 2048 * 1024
