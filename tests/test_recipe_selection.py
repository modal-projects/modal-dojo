import base64
import dataclasses
import importlib
import inspect
import pkgutil
from typing import Any

import pytest

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_utils import (
    get_checkpoint_conversion_policy,
    prepare_launch_config,
)
from modal_training_gym.common.models import Qwen3_4B
from modal_training_gym.common.models.validation import Framework, _ValidationConfig
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.gpu_allocation import (
    validate_megatron_actor_parallelism,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.gemma4_26b_a4b import (
    Gemma4_26B_A4B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.inkling import Inkling_Small_Recipe
from modal_training_gym.train_recipes.miles_recipe.moonlight_16b_a3b import (
    Moonlight_16B_A3B_Recipe,
)
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4B_Miles_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4B_Recipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_5_0_8b import (
    Qwen3_5_0_8B_Recipe,
)

_RECIPE_PACKAGES = (
    "modal_training_gym.train_recipes.slime_recipe",
    "modal_training_gym.train_recipes.miles_recipe",
)

_BASE_RECIPE = {
    Framework.SLIME: SlimeRecipe.get_base_recipe,
    Framework.MILES: MilesRecipe.get_base_recipe,
}


def _dataset() -> HuggingFaceDataset:
    return HuggingFaceDataset(
        hf_repo="some/dataset",
        input_column="prompt",
        output_column="answer",
        input_format="text",
    )


def _config(recipe: SlimeRecipe) -> TrainConfig:
    return TrainConfig(dataset=_dataset(), model=Qwen3_4B(), recipe=recipe)


def _model_recipe_classes() -> list[type[SlimeRecipe | MilesRecipe]]:
    classes = []
    for package_name in _RECIPE_PACKAGES:
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__, f"{package_name}."):
            if module_info.name.endswith(".recipe"):
                continue
            module = importlib.import_module(module_info.name)
            classes.extend(
                recipe_cls
                for _, recipe_cls in inspect.getmembers(module, inspect.isclass)
                if recipe_cls.__module__ == module.__name__
                and "Recipe" in recipe_cls.__name__
                and issubclass(recipe_cls, (SlimeRecipe, MilesRecipe))
            )
    return classes


@pytest.mark.parametrize(
    "recipe_cls",
    _model_recipe_classes(),
    ids=lambda recipe_cls: recipe_cls.__name__,
)
def test_model_recipes_only_override_framework_defaults(
    recipe_cls: type[SlimeRecipe | MilesRecipe],
) -> None:
    baseline_cls = recipe_cls.__mro__[1]  # mro[1] corresponds to the parent class
    defaults = {}
    for field in dataclasses.fields(baseline_cls):
        if field.default is dataclasses.MISSING:
            if field.default_factory is dataclasses.MISSING:
                continue
            defaults[field.name] = field.default_factory()
        else:
            defaults[field.name] = field.default
    recipe = recipe_cls()
    direct_fields = set(recipe_cls.__annotations__) & defaults.keys()

    redundant = {
        field for field in direct_fields if getattr(recipe, field) == defaults[field]
    }
    assert redundant == set()


def test_gemma_recipe_disables_unsupported_recompute_and_routing_replay() -> None:
    recipe = Gemma4_26B_A4B_Recipe()

    # Recompute rejects Gemma's tuple decoder output. Routing replay expects
    # num_experts_per_tok, which Gemma names top_k_experts.
    assert recipe.recompute_granularity is None
    assert recipe.recompute_method is None
    assert recipe.recompute_num_layers is None
    assert recipe.use_rollout_routing_replay is False


@pytest.mark.parametrize(
    "config",
    _ValidationConfig.select(),
    ids=lambda config: config.name,
)
def test_registered_recipe_batch_sizes_divide_data_parallel(
    config: _ValidationConfig,
) -> None:
    recipe = _BASE_RECIPE[config.framework](config.model_config())
    assert recipe is not None
    validate_megatron_actor_parallelism(recipe)

    world = recipe.actor_num_nodes * recipe.actor_num_gpus_per_node
    model_parallel = (
        (getattr(recipe, "tensor_model_parallel_size", 1) or 1)
        * (getattr(recipe, "pipeline_model_parallel_size", 1) or 1)
        * (getattr(recipe, "context_parallel_size", 1) or 1)
    )
    dp = world // model_parallel
    micro = getattr(recipe, "micro_batch_size", None) or 1
    global_batch_size = getattr(recipe, "global_batch_size", None)
    samples = recipe.rollout_batch_size * recipe.n_samples_per_prompt
    if global_batch_size:
        assert global_batch_size % (micro * dp) == 0
        assert samples % global_batch_size == 0

    if isinstance(recipe, MilesRecipe):
        convert_nodes, convert_nproc, _ = get_checkpoint_conversion_policy(recipe)
        assert convert_nodes * convert_nproc <= world


def test_slime_recipe_is_constructible_without_kwargs() -> None:
    SlimeRecipe()
    assert SlimeRecipe(num_rollout=7)._fields()["save_interval"] == 7
    assert "--save-interval" not in SlimeRecipe(save=None).cli_args()
    assert "--save-interval" in SlimeRecipe().cli_args()


def test_generic_recipe_uses_framework_defaults_for_known_model() -> None:
    config = _config(SlimeRecipe())

    recipe = config._prepare_recipe()
    assert recipe.gpu_type == "H100"
    assert recipe.tensor_model_parallel_size == 1
    assert recipe.rollout_num_gpus_per_engine == 1
    assert recipe.colocate is True
    assert recipe.num_rollout == 1
    assert recipe.n_samples_per_prompt == 2
    assert recipe.rollout_batch_size == 2


def test_miles_recipe_uses_shared_sampling_defaults() -> None:
    recipe = MilesRecipe()

    assert recipe.n_samples_per_prompt == 2
    assert MilesRecipe(num_rollout=7)._fields()["save_interval"] == 7
    assert recipe.rollout_batch_size == 2


@pytest.mark.parametrize("recipe_cls", [SlimeRecipe, MilesRecipe])
def test_save_interval_follows_extra_config_num_rollout(recipe_cls, tmp_path) -> None:
    recipe = recipe_cls(num_rollout=1, extra_config={"num_rollout": 10})
    assert recipe._fields()["save_interval"] == 10

    prepare_launch_config(
        recipe, None, str(tmp_path), yaml_config_fields=("extra_config",)
    )
    assert recipe._fields()["save_interval"] == 10


def _inkling_image_patch_sources(recipe: MilesRecipe) -> list[str]:
    sources = []
    for cmd in recipe.image_run_commands:
        parts = cmd.split()
        if len(parts) >= 2 and "base64" in cmd:
            sources.append(base64.b64decode(parts[1]).decode())
    return sources


def test_inkling_patches_router_startup_timeout() -> None:
    recipe = Inkling_Small_Recipe(image_run_commands=["echo extra"])
    sources = _inkling_image_patch_sources(recipe)
    assert any("PATCHED_ROUTER_STARTUP_TIMEOUT" in src for src in sources)
    assert any("PATCHED_QKVR_CPU_MERGE" in src for src in sources)
    assert "echo extra" in recipe.image_run_commands


def test_model_recipe_uses_its_class_defaults() -> None:
    recipe = _config(Qwen3_4B_Recipe())._prepare_recipe()

    assert recipe.actor_num_gpus_per_node == 1
    assert recipe.max_tokens_per_gpu == 8192


def test_prepare_recipe_does_not_mutate_stored_launch_callables() -> None:
    def image_overlay(image: Any) -> Any:
        return image

    def custom_rm_function(*args: Any, **kwargs: Any) -> float:
        return 1.0

    recipe = SlimeRecipe(
        image_overlay=image_overlay,
        custom_rm_function=custom_rm_function,
    )
    config = _config(recipe)

    first = config._prepare_recipe()
    first.image_overlay = None
    first.custom_rm_function = None
    second = config._prepare_recipe()

    assert recipe.image_overlay is image_overlay
    assert recipe.custom_rm_function is custom_rm_function
    assert second.image_overlay is image_overlay
    assert second.custom_rm_function is custom_rm_function


@pytest.mark.parametrize(
    ("recipe_cls", "framework"), [(SlimeRecipe, "slime"), (MilesRecipe, "miles")]
)
def test_sft_loss_emits_native_sft_flags(recipe_cls, framework) -> None:
    recipe = recipe_cls(global_batch_size=8, n_samples_per_prompt=4, num_epoch=3)
    assert "--loss-type" not in recipe.cli_args(dataset=_dataset())
    recipe.loss_type = "sft_loss"
    args = recipe.cli_args(dataset=_dataset())
    values = dict(zip(args, args[1:]))
    assert values["--loss-type"] == "sft_loss"
    assert values["--rollout-function-path"] == (
        f"{framework}.rollout.sft_rollout.generate_rollout"
    )
    assert values["--n-samples-per-prompt"] == "1"
    assert values["--rollout-batch-size"] == "8"
    assert "--loss-mask-type" not in args
    assert {"--debug-train-only", "--disable-compute-advantages-and-returns"} <= set(
        args
    )
    assert not {"--apply-chat-template", "--num-rollout", "--colocate"} & set(args)
    assert recipe.gpu_allocation.rollout_gpus == 0
    assert recipe.train_async is (recipe_cls is MilesRecipe)


def test_sft_qwen35_emits_loss_mask_type_qwen3_5() -> None:
    assert "--loss-mask-type" not in Qwen3_5_0_8B_Recipe().cli_args(dataset=_dataset())
    sft_args = Qwen3_5_0_8B_Recipe(loss_type="sft_loss").cli_args(dataset=_dataset())
    assert dict(zip(sft_args, sft_args[1:]))["--loss-mask-type"] == "qwen3_5"


def test_sft_none_global_batch_size_uses_rollout_batch_size() -> None:
    recipe = Moonlight_16B_A3B_Recipe(loss_type="sft_loss")
    assert recipe.global_batch_size is None
    args = recipe.cli_args(dataset=_dataset())
    values = dict(zip(args, args[1:]))
    assert values["--global-batch-size"] == str(recipe.rollout_batch_size)
    assert values["--rollout-batch-size"] == str(recipe.rollout_batch_size)


def test_miles_qwen35_sft_raises() -> None:
    recipe = Qwen3_5_4B_Miles_Recipe(loss_type="sft_loss")
    with pytest.raises(TrainingGymConfigError, match="qwen3_5"):
        recipe.cli_args(dataset=_dataset())
