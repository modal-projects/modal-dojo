"""MilesRecipe hook stashing: the four intercepted log/megatron hooks ride
inside ``extra_config`` under ``training_gym_*`` keys. ``extra_config`` is
dict-only (matching ``SlimeRecipe``) so the keys are always stashable.
"""

from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.phase_reporting import _hook_path_from_args
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe


def test_str_hook_is_stashed_in_extra_config() -> None:
    recipe = MilesRecipe(custom_rollout_log_function="my_pkg.hooks.log_fn")
    assert isinstance(recipe.extra_config, dict)
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == "my_pkg.hooks.log_fn"
    )


def test_yaml_path_extra_config_is_rejected() -> None:
    # dict-only, matching SlimeRecipe: a YAML *path* can't stash the
    # training_gym_* hook keys and its native keys would clobber the wrapper
    # CLI flags at runtime.
    with pytest.raises(ValueError, match="extra_config"):
        MilesRecipe(extra_config="configs/extra.yaml")


def test_native_key_in_extra_config_is_migrated() -> None:
    # miles applies custom-config YAML keys onto args after argparse, so a
    # native key left in extra_config would clobber the wrapper CLI flag.
    recipe = MilesRecipe(
        extra_config={"custom_rollout_log_function_path": "my_pkg.hooks.log_fn"}
    )
    assert "custom_rollout_log_function_path" not in recipe.extra_config
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == "my_pkg.hooks.log_fn"
    )


def test_native_key_dropped_when_training_gym_key_present() -> None:
    recipe = MilesRecipe(
        extra_config={
            "custom_rollout_log_function_path": "old.fn",
            "training_gym_custom_rollout_log_function_path": "my_pkg.hooks.log_fn",
        }
    )
    assert "custom_rollout_log_function_path" not in recipe.extra_config
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == "my_pkg.hooks.log_fn"
    )


def test_hook_field_wins_over_native_key() -> None:
    recipe = MilesRecipe(
        custom_rollout_log_function="field.fn",
        extra_config={"custom_rollout_log_function_path": "native.fn"},
    )
    assert "custom_rollout_log_function_path" not in recipe.extra_config
    assert (
        recipe.extra_config["training_gym_custom_rollout_log_function_path"]
        == "field.fn"
    )


_KEY = "training_gym_custom_rollout_log_function_path"


def test_hook_lookup_prefers_training_gym_key() -> None:
    args = SimpleNamespace(
        extra_config={
            _KEY: "my_pkg.hooks.log_fn",
            "custom_rollout_log_function_path": "other.fn",
        }
    )
    assert _hook_path_from_args(args, _KEY) == "my_pkg.hooks.log_fn"


def test_hook_lookup_falls_back_to_native_key() -> None:
    args = SimpleNamespace(
        extra_config={"custom_rollout_log_function_path": "my_pkg.hooks.log_fn"}
    )
    assert _hook_path_from_args(args, _KEY) == "my_pkg.hooks.log_fn"


def test_hook_lookup_never_dispatches_to_gym_wrapper() -> None:
    args = SimpleNamespace(
        extra_config={
            "custom_rollout_log_function_path": (
                "modal_dojo.frameworks.miles.phase_reporting.log_rollout_data"
            )
        }
    )
    assert _hook_path_from_args(args, _KEY) is None
