"""DeepSeek-V4.1 recipe context-parallelism validation."""

import warnings

import pytest

from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash import (
    DeepSeek_V4_1_Flash_Recipe,
)


def _recipe(**overrides) -> DeepSeek_V4_1_Flash_Recipe:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DeepSeek_V4_1_Flash_Recipe(**overrides)


def test_context_parallel_enables_allgather_cp() -> None:
    recipe = _recipe(context_parallel_size=2)
    assert recipe.extra_config["allgather_cp"] is True


def test_explicit_zigzag_cp_is_rejected() -> None:
    with pytest.raises(ValueError, match="allgather_cp"):
        _recipe(context_parallel_size=2, extra_config={"allgather_cp": False})


def test_no_context_parallelism_leaves_allgather_unset() -> None:
    recipe = _recipe()
    assert "allgather_cp" not in (recipe.extra_config or {})
