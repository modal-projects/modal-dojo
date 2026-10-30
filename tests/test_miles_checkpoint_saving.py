import asyncio
from types import SimpleNamespace

import pytest
import yaml

from modal_dojo.common.framework import Framework
from modal_dojo.common.launcher_helpers import configured_recipe_save
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.run import TrainingRun, set_checkpoint_location
from modal_dojo.frameworks.miles.modal_helpers.utils import prepare_miles_config
from modal_dojo.train_recipes.miles_recipe import MilesRecipe


@pytest.mark.parametrize("instrumented", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("fully_async", [False, True])
def test_beam_prefetch_wait_is_outside_checkpoint_timing(
    monkeypatch, instrumented, pending, fully_async
):
    from modal_dojo.common import timing_recorder as timing
    from modal_dojo.common.derived_metrics import driver_step_seconds
    from modal_dojo.frameworks.miles.modal_helpers.patches.patch_beam_checkpoint import (
        patch_source,
    )

    source = """async def save(args, rollout_executor, rollout_data_next_future):
    rollout_id = 1
    await rollout_executor.save(rollout_id)
    return rollout_data_next_future
"""
    if instrumented:
        source = source.replace(
            "    await rollout_executor.save(rollout_id)",
            "    with _tg_time_phase('checkpoint_save'):\n"
            "        await rollout_executor.save(rollout_id)",
        )
    patched = patch_source(source, "train_async.py")
    assert patch_source(patched, "train_async.py") == patched
    env = {"_tg_time_phase": timing.time_phase}
    exec(compile(patched, "train_async.py", "exec"), env)

    clock = [0.0]
    monkeypatch.setenv("MODAL_DOJO_SUBSTEP_TIMING", "on")
    monkeypatch.setattr(timing.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(timing.RoleRecorder, "_publish", lambda *a, **kw: None)

    class Future:
        consumed = False

        def __await__(self):
            clock[0] += 600.0
            self.consumed = True
            return iter(())

    future = Future() if pending else None
    should_wait = pending and not fully_async

    class Executor:
        async def save(self, rollout_id):
            assert rollout_id == 1
            assert not should_wait or future.consumed
            clock[0] += 5.0

    with timing.recording_lane("driver", 1) as recorder:
        with timing.time_phase("train_models"):
            clock[0] += 10.0
        result = asyncio.run(
            env["save"](SimpleNamespace(fully_async=fully_async), Executor(), future)
        )
    assert result is future  # Normal loop consumption must still receive the batch.
    phases = recorder.phases
    if should_wait:
        assert phases["wait_for_next_rollout"]["busy_duration_s"] == 600.0
    else:
        assert "wait_for_next_rollout" not in phases
    if instrumented:
        assert phases["checkpoint_save"]["busy_duration_s"] == 5.0
    measured = driver_step_seconds({"1": {"roles": {"driver": {"phases": phases}}}})
    assert measured == {2: 610.0 if should_wait else 10.0}


def test_explicitly_disabled_saving_omits_both_flags():
    recipe = MilesRecipe(save_interval=None, save=None, ref_load="/checkpoints/initial")
    fields = recipe._fields()
    assert fields["save"] is None
    assert fields.get("save_interval") is None
    assert "--save" not in recipe.cli_args()
    assert "--save-interval" not in recipe.cli_args()
    assert fields["ref_load"] == "/checkpoints/initial"
    assert recipe.save is None


def test_enabled_saving_keeps_path_and_interval():
    fields = MilesRecipe(save_interval=10, save="/checkpoints/run")._fields()
    assert fields["save"] == "/checkpoints/run"
    assert fields["save_interval"] == 10


@pytest.mark.parametrize("enabled", [False, True])
def test_save_overrides_survive_yaml_materialization(tmp_path, enabled):
    override = {
        "save": "/checkpoints/override" if enabled else None,
        "save_interval": 10 if enabled else None,
    }
    recipe = MilesRecipe(
        save=None if enabled else "/checkpoints/original",
        save_interval=None if enabled else 10,
        extra_config=override,
    )
    for materialized in (False, True):
        if materialized:
            prepare_miles_config(recipe, None, str(tmp_path))
            with open(recipe.extra_config) as f:
                assert yaml.safe_load(f) == override
        args = recipe.cli_args()
        if enabled:
            assert args[args.index("--save") + 1] == "/checkpoints/override"
            assert args[args.index("--save-interval") + 1] == "10"
        else:
            assert "--save" not in args
            assert "--save-interval" not in args


def test_run_without_checkpoint_keeps_original_model_source(fake_volume):
    model = ModelConfig(model_name="org/model", model_path="/hf/model")
    run = TrainingRun(
        training_run_id="run",
        framework=Framework.MILES,
        config={},
        app_name="test",
        source_model=model,
    )
    assert run.checkpoints() == []
    assert run.model.model_path == "/hf/model"


def test_disabled_save_does_not_record_a_checkpoint_dir() -> None:
    recipe = MilesRecipe(save=None, save_interval=None)

    assert configured_recipe_save(recipe) is None


def test_empty_recorded_location_keeps_source_model(fake_volume):
    model = ModelConfig(model_name="org/model", model_path="/hf/model")
    run = TrainingRun(
        training_run_id="run",
        framework=Framework.MILES,
        config={},
        app_name="test",
        source_model=model,
    )
    set_checkpoint_location(
        run,
        checkpoint_dir="",
        checkpoints_volume_name="miles-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )

    assert run.checkpoint_dir == ""
    assert run.model.model_path == "/hf/model"
