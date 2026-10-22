from __future__ import annotations

import inspect

import pytest

from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
from modal_dojo.common.dataset import HuggingFaceDataset
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.launcher_helpers import resumed_recipe
from modal_dojo.common.models import Qwen3_5_4B
from modal_dojo.common.train import TrainConfig
from modal_dojo.frameworks.miles.launcher import build_miles_app
from modal_dojo.frameworks.slime.launcher import build_slime_app
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

_RECIPE_KW = dict(
    gpu_type="H100",
    colocate=True,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=1,
    num_rollout=1,
    rollout_batch_size=16,
    rollout_max_response_len=4096,
    rollout_temperature=1.0,
    save_interval=10,
)


def _checkpoint(checkpoint_type: CheckpointType) -> Checkpoint:
    suffix = "_hf" if checkpoint_type == CheckpointType.hf else ""
    return Checkpoint(
        checkpoint_type=checkpoint_type,
        name=f"iter_0000009{suffix}",
        path=f"/checkpoints/run/iter_0000009{suffix}",
        timestamp=1.0,
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _config(recipe, checkpoint_type: CheckpointType | None) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_5_4B(),
        dataset=HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
            input_format="text",
        ),
        recipe=recipe,
        resume_from_checkpoint=None
        if checkpoint_type is None
        else _checkpoint(checkpoint_type),
    )


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(SlimeRecipe(**_RECIPE_KW), id="slime"),
        pytest.param(MilesRecipe(), id="miles"),
    ],
)
def test_megatron_resume_starts_new_run_from_weights(recipe) -> None:
    config = _config(recipe, CheckpointType.megatron)
    prepared = config._prepare_recipe()
    fields = prepared._fields(model=config.model)
    args = prepared.cli_args(model=config.model)

    assert prepared is not recipe
    assert prepared.load == "/checkpoints/run"
    assert prepared.start_rollout_id == 0
    assert prepared.no_load_optim is True
    assert args[args.index("--start-rollout-id") + 1] == "0"
    assert "--no-load-optim" in args
    assert fields["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert config.model.model_path is None
    assert recipe.load == ""
    assert recipe.start_rollout_id is None
    assert "--start-rollout-id" not in recipe.cli_args(model=config.model)


def test_explicit_start_rollout_id_wins_over_resume_from_checkpoint_default() -> None:
    config = _config(
        SlimeRecipe(**_RECIPE_KW, start_rollout_id=5), CheckpointType.megatron
    )
    prepared = config._prepare_recipe()

    assert prepared.start_rollout_id == 5
    assert prepared.no_load_optim is True


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(SlimeRecipe(**_RECIPE_KW, load="/checkpoints/run"), id="slime"),
        pytest.param(MilesRecipe(load="/checkpoints/run"), id="miles"),
    ],
)
def test_recipe_load_without_resume_from_checkpoint_continues_with_adam(recipe) -> None:
    config = _config(recipe, None)
    prepared = config._prepare_recipe()
    args = prepared.cli_args(model=config.model)

    assert prepared.load == "/checkpoints/run"
    assert prepared.start_rollout_id is None
    assert prepared.no_load_optim is False
    assert "--start-rollout-id" not in args
    assert "--no-load-optim" not in args


def test_resume_from_checkpoint_wins_over_recipe_load() -> None:
    config = _config(
        SlimeRecipe(**_RECIPE_KW, load="/checkpoints/other"),
        CheckpointType.megatron,
    )

    assert config._prepare_recipe().load == "/checkpoints/run"


def test_config_summary_records_resume_from_checkpoint() -> None:
    config = _config(SlimeRecipe(**_RECIPE_KW), CheckpointType.megatron)

    summary = config._build_config_summary("run-id")

    assert summary["recipe"]["load"] == "/checkpoints/run"
    assert summary["recipe"]["hf_checkpoint"] == "Qwen/Qwen3.5-4B"
    assert config.recipe.load == ""
    assert config.model.model_path is None


def test_hf_export_is_not_a_training_resume_checkpoint() -> None:
    config = _config(SlimeRecipe(**_RECIPE_KW), CheckpointType.hf)

    with pytest.raises(
        DojoConfigError,
        match="Hugging Face exports are serving artifacts",
    ):
        config._prepare_recipe()


def test_launchers_do_not_replace_model_path_with_checkpoint() -> None:
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_slime_app
    )
    assert "model.model_path = checkpoint.path" not in inspect.getsource(
        build_miles_app
    )


def test_slime_conversion_uses_wrapper_with_expected_environment() -> None:
    source = inspect.getsource(build_slime_app)

    assert (
        "modal_dojo.frameworks.slime.modal_helpers.convert_hf_to_torch_dist" in source
    )
    assert (
        'convert_script = f"{SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"'
        not in source
    )
    assert (
        'if any(arg.startswith("--pipeline-model-parallel-size ") '
        "for arg in extra_args):\n"
        '            env["SKIP_PP_AUTOINFLATE"] = "1"'
    ) in source
    assert 'if num_nodes > 1:\n            env["SKIP_RELEASE_RENAME"] = "1"' in source


@pytest.mark.parametrize("recipe_cls", [SlimeRecipe, MilesRecipe])
@pytest.mark.parametrize("no_save_optim", [False, True])
def test_internal_resume_uses_saved_optimizer_and_restores_recipe(
    recipe_cls, no_save_optim
) -> None:
    from modal_dojo.common.launcher_helpers import resumed_recipe

    recipe = recipe_cls(
        num_rollout=10,
        load="/checkpoints/seed",
        start_rollout_id=0,
        no_load_optim=not no_save_optim,
        no_save_optim=no_save_optim,
    )
    checkpoint = {
        "resume_from_iteration": 2,
        "resume_checkpoint_path": "/checkpoints/run/iter_0000002",
    }
    with pytest.raises(RuntimeError, match="command failed"):
        with resumed_recipe(recipe, "/checkpoints/run", checkpoint):
            fields = recipe._fields()
            assert fields["load"] == "/checkpoints/run"
            assert fields["start_rollout_id"] is None
            assert fields["no_load_optim"] is no_save_optim
            raise RuntimeError("command failed")

    assert recipe.load == "/checkpoints/seed"
    assert recipe.start_rollout_id == 0
    assert recipe.no_load_optim is not no_save_optim


@pytest.mark.parametrize("recipe_cls", [SlimeRecipe, MilesRecipe])
def test_internal_resume_after_epoch_save_under_num_epoch(recipe_cls) -> None:
    recipe = recipe_cls(num_rollout=1, extra_config={"num_epoch": 3})
    checkpoint = {
        "resume_from_iteration": 5,
        "resume_checkpoint_path": "/checkpoints/run/iter_0000005",
    }
    with resumed_recipe(recipe, "/checkpoints/run", checkpoint):
        assert recipe.load == "/checkpoints/run"


def test_miles_conversion_uses_wrapper_with_expected_environment() -> None:
    source = inspect.getsource(build_miles_app)

    assert (
        "modal_dojo.frameworks.miles.modal_helpers.convert_hf_to_torch_dist" in source
    )
    assert (
        'convert_script = f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"'
        not in source
    )
    assert (
        'if any(arg.startswith("--pipeline-model-parallel-size ") '
        "for arg in extra_args):\n"
        '            env["CONVERT_KEEP_PP1"] = "1"'
    ) in source
    assert 'if num_nodes > 1:\n            env["SKIP_RELEASE_RENAME"] = "1"' in source


@pytest.mark.parametrize(
    "recipe",
    [
        pytest.param(SlimeRecipe(**_RECIPE_KW), id="slime"),
        pytest.param(MilesRecipe(), id="miles"),
    ],
)
def test_auto_resume_drops_extra_config_start_rollout_id(recipe, tmp_path) -> None:
    import yaml

    from modal_dojo.common.launcher_utils import (
        prepare_launch_config,
    )

    from modal_dojo.common.launcher_helpers import resumed_recipe

    recipe.extra_config = {"start_rollout_id": 0, "qkv_format": "bshd"}
    prepare_launch_config(
        recipe, None, str(tmp_path), yaml_config_fields=("extra_config",)
    )
    assert "start_rollout_id" in recipe._escape_hatch_keys()

    with resumed_recipe(
        recipe,
        "/checkpoints/run",
        {"resume_checkpoint_path": "/checkpoints/run/iter_0000000"},
    ):
        assert recipe.start_rollout_id is None

    assert recipe._escape_hatch_keys() == ("qkv_format",)
    with open(recipe.extra_config) as f:
        assert yaml.safe_load(f) == {"qkv_format": "bshd"}
