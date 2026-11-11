import pytest

from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


@pytest.mark.parametrize("recipe_cls", [MilesRecipe, SlimeRecipe])
def test_over_sampling_below_rollout_batch_size_is_rejected(recipe_cls) -> None:
    with pytest.raises(ValueError, match="over_sampling_batch_size=2 must be >="):
        recipe_cls(rollout_batch_size=4, over_sampling_batch_size=2)


@pytest.mark.parametrize("recipe_cls", [MilesRecipe, SlimeRecipe])
@pytest.mark.parametrize("over_sampling", [None, 4, 8])
def test_over_sampling_at_or_above_rollout_batch_size_is_accepted(
    recipe_cls, over_sampling
) -> None:
    recipe = recipe_cls(rollout_batch_size=4, over_sampling_batch_size=over_sampling)
    assert recipe.over_sampling_batch_size == over_sampling
