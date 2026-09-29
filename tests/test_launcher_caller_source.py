from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from modal_dojo.common.launcher_helpers import (
    mount_caller_source,
    ship_callable,
)
from modal_dojo.common.launcher_helpers import ship_recipe_callables
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


_WRAPPED_HOOKS = (
    "custom_rollout_log_function",
    "custom_megatron_before_log_prob_hook",
    "custom_megatron_before_train_step_hook",
)


class RecordingImage:
    def __init__(self) -> None:
        self.operations: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def add_local_file(self, *args: object, **kwargs: object) -> RecordingImage:
        self.operations.append(("add_local_file", args, kwargs))
        return self


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mount_caller_source_flattens_flat_script(tmp_path: Path) -> None:
    script = tmp_path / "flat.py"
    script.write_text("VALUE = 1\n")

    image = mount_caller_source(RecordingImage(), str(script))

    assert image.operations == [
        (
            "add_local_file",
            (str(script),),
            {"remote_path": "/root/flat.py", "copy": True},
        )
    ]


def test_mount_caller_source_skips_when_caller_script_is_none() -> None:
    image = mount_caller_source(RecordingImage(), None)

    assert image.operations == []


def test_ship_callable_flattens_sibling_file(tmp_path: Path) -> None:
    caller = tmp_path / "caller.py"
    caller.write_text("")
    helper = tmp_path / "reward.py"
    helper.write_text("def score() -> int:\n    return 1\n")
    module = _load_module(helper, "reward")

    image, path = ship_callable(
        RecordingImage(),
        module.score,
        caller_script=str(caller),
        fallback_name="score",
    )

    assert path == "reward.score"
    assert image.operations == [
        (
            "add_local_file",
            (str(helper),),
            {"remote_path": "/root/reward.py", "copy": True},
        )
    ]


@pytest.mark.parametrize("recipe_cls", [SlimeRecipe, MilesRecipe])
def test_recipe_hooks_ship_closures_to_framework_destinations(recipe_cls):
    captured = "closure value"

    def hook():
        return captured

    recipe = recipe_cls(
        custom_rm_function=hook,
        custom_generate_function=hook,
        custom_reward_post_process_function=hook,
        rollout_function=hook,
        custom_rollout_log_function=hook,
        custom_eval_rollout_log_function="user.eval_hook",
        custom_megatron_before_log_prob_hook=hook,
        custom_megatron_before_train_step_hook=hook,
    )
    image = ship_recipe_callables(
        RecordingImage(),
        recipe,
        caller_script=str(Path(__file__).resolve()),
        reward_post_process_in_config=recipe_cls is SlimeRecipe,
    )

    shipped = set()
    for _, (local_path,), options in image.operations:
        source = Path(local_path)
        namespace: dict = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        assert namespace["hook"]() == captured  # closures ship by value
        shipped.add(f"{Path(options['remote_path']).stem}.hook")
        source.unlink()

    config, fields = recipe.extra_config, recipe._fields()
    reward_key = "custom_reward_post_process_path"
    destinations = [
        config["custom_rm_path"],
        config["custom_generate_function_path"],
        fields["rollout_function_path"],
        config[reward_key] if recipe_cls is SlimeRecipe else fields[reward_key],
        *(
            config[f"training_gym_{name}_path"]
            for name in _WRAPPED_HOOKS
            if getattr(recipe, name) is None
        ),
    ]
    assert shipped == set(destinations)
    assert all(".phase_reporting." in fields[f"{n}_path"] for n in _WRAPPED_HOOKS)
    assert config["training_gym_custom_eval_rollout_log_function_path"] == (
        "user.eval_hook"
    )
