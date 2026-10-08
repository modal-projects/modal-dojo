from types import SimpleNamespace

from modal_dojo.common.launcher_helpers import (
    experimental_options,
    training_function_options,
)


def _recipe(**train_function_kwargs):
    return SimpleNamespace(
        train_function_kwargs=train_function_kwargs,
        gpu_type="B300",
        gpu_allocation=SimpleNamespace(gpus_per_node=8),
        memory=None,
        cpu=None,
        cloud=None,
        region=None,
        max_retries=0,
    )


def test_no_efa_option_is_set_by_default():
    assert experimental_options(_recipe()) == {}
    options = training_function_options(
        _recipe(), framework="miles", secrets=[], experimental_options={}
    )
    assert options["experimental_options"] == {}


def test_recipe_can_disable_efa():
    recipe = _recipe(experimental_options={"efa_disabled": True})
    assert experimental_options(recipe) == {"efa_disabled": True}
    options = training_function_options(
        recipe, framework="miles", secrets=[], experimental_options={}
    )
    assert options["experimental_options"] == {"efa_disabled": True}


def test_none_train_function_kwargs():
    assert experimental_options(SimpleNamespace(train_function_kwargs=None)) == {}


def test_high_priority_dropped_for_non_clustered_functions():
    recipe = _recipe(experimental_options={"efa_disabled": True, "priority": "high"})
    assert experimental_options(recipe) == {"efa_disabled": True, "priority": "high"}
    assert experimental_options(recipe, clustered=False) == {"efa_disabled": True}

    legacy = _recipe(experimental_options={"high_priority": True})
    assert experimental_options(legacy, clustered=False) == {}


def test_low_priority_kept_for_non_clustered_functions():
    recipe = _recipe(experimental_options={"priority": "low"})
    assert experimental_options(recipe, clustered=False) == {"priority": "low"}
