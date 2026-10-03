"""MilesRecipe config checks that surface at launch instead of mid-run."""

import warnings

import pytest

from modal_dojo.train_recipes.miles_recipe.glm_5_3_flash_lora import (
    GLM_5_3_Flash_LoRA_Recipe,
)
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


def test_routing_replay_with_custom_rollout_warns() -> None:
    with pytest.warns(UserWarning, match="return_routed_experts"):
        MilesRecipe(
            use_rollout_routing_replay=True,
            rollout_function="my_pkg.rollout.generate_rollout",
        )


def test_routing_replay_with_builtin_rollout_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MilesRecipe(use_rollout_routing_replay=True)


def test_glm_5_3_flash_defaults_fit_three_dp_ranks() -> None:
    recipe = GLM_5_3_Flash_LoRA_Recipe()
    assert recipe.use_dynamic_batch_size
    assert recipe.global_batch_size % 3 == 0
    assert (
        recipe.rollout_batch_size * recipe.n_samples_per_prompt
        == recipe.global_batch_size
    )
