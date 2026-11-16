from types import SimpleNamespace

import pytest

from modal_dojo import (
    Budget,
    DatasetConfig,
    Environment,
    Fallback,
    ManagedTrainConfig,
    Qwen3_5_4B,
    Qwen3_5_4B_Recipe,
)
from modal_dojo.common import fallback as fb
from modal_dojo.common import managed_train
from modal_dojo.common.errors import DojoConfigError


class _Rows(DatasetConfig):
    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def rows(self):
        return [{"prompt": "q", "label": "a"}]


async def reward(args, sample, **kwargs) -> float:
    return 0.0


def test_fallback_needs_recipes_and_a_threshold():
    with pytest.raises(DojoConfigError, match="at least one"):
        Fallback([], threshold=60)
    with pytest.raises(DojoConfigError, match="positive"):
        Fallback([Qwen3_5_4B_Recipe()], threshold=0)


def test_wait_for_training_start_polls_until_started_or_timeout():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    started_at = {"run": 40.0}
    started = lambda run_id: now[0] >= started_at.get(run_id, float("inf"))  # noqa: E731

    assert fb.wait_for_training_start(
        "run", 3600, started=started, sleep=sleep, clock=lambda: now[0]
    )
    assert sleeps == [15.0, 15.0, 15.0]

    now[0], sleeps[:] = 0.0, []
    assert not fb.wait_for_training_start(
        "never", 20, started=started, sleep=sleep, clock=lambda: now[0]
    )
    assert sleeps == [15.0, 5.0]


def _config(gpu: str):
    recipe = SimpleNamespace(gpu_type=gpu, gpu_allocation=SimpleNamespace(total_gpus=8))
    return SimpleNamespace(recipe=recipe, name=gpu)


def _run_fallback(configs, starts):
    events = []

    def launch(config):
        events.append(("launch", config.name))
        return SimpleNamespace(training_run_id=f"run-{config.name}")

    def wait(run_id, threshold):
        events.append(("wait", run_id, threshold))
        return run_id in starts

    def abandon(run):
        events.append(("abandon", run.training_run_id))

    run = fb.launch_with_fallback(
        configs, 3600, launch=launch, wait=wait, abandon=abandon
    )
    return run, events


def test_first_recipe_that_starts_wins():
    configs = [_config("H200"), _config("H100"), _config("B200")]
    run, events = _run_fallback(configs, starts={"run-H200"})
    assert run.training_run_id == "run-H200"
    assert events == [("launch", "H200"), ("wait", "run-H200", 3600)]


def test_falls_back_and_leaves_the_last_recipe_queued():
    configs = [_config("H200"), _config("H100"), _config("B200")]
    run, events = _run_fallback(configs, starts=set())
    assert run.training_run_id == "run-B200"
    assert events == [
        ("launch", "H200"),
        ("wait", "run-H200", 3600),
        ("abandon", "run-H200"),
        ("launch", "H100"),
        ("wait", "run-H100", 3600),
        ("abandon", "run-H100"),
        ("launch", "B200"),  # the last is never abandoned
    ]


def test_managed_config_builds_each_fallback_recipe():
    env = Environment(_Rows(), reward)
    budget = Budget(max_hours=4)
    fallback = Fallback(
        [Qwen3_5_4B_Recipe(gpu_type="H200"), Qwen3_5_4B_Recipe(gpu_type="H100")],
        threshold=3600,
    )
    managed = ManagedTrainConfig(
        Qwen3_5_4B(),
        env,
        budget,
        recipe_overrides={"lr": 1e-6},
        fallback_recipes=fallback,
    )
    recipes = [config.recipe for config in managed.train_configs]
    assert [r.gpu_type for r in recipes] == ["H200", "H100"]
    for recipe in recipes:
        assert recipe.budget is budget
        assert recipe.custom_rm_function is reward
        assert recipe.lr == 1e-6
        assert recipe.rollout_batch_size == 1  # capped to the one task
    assert managed.train_config is managed.train_configs[0]


def test_managed_launch_uses_fallback_only_when_given(monkeypatch):
    env = Environment(_Rows(), reward)
    calls = []
    monkeypatch.setattr(
        managed_train,
        "launch_with_fallback",
        lambda configs, threshold, launch: calls.append((len(configs), threshold)),
    )
    fallback = Fallback(
        [Qwen3_5_4B_Recipe(gpu_type="H200"), Qwen3_5_4B_Recipe()], threshold=60
    )
    ManagedTrainConfig(
        Qwen3_5_4B(), env, Budget(max_rollouts=2), fallback_recipes=fallback
    ).launch()
    assert calls == [(2, 60)]

    single = ManagedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=2))
    assert len(single.train_configs) == 1
