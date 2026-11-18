import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo import Budget, Qwen3_5_4B_Recipe
from modal_dojo.common import budget as budget_mod
from modal_dojo.common.budget_stop_patch import MARKER, patch_source
from modal_dojo.common.errors import DojoConfigError

TESTDATA = Path(__file__).parent / "testdata" / "budget_stop"


# ── Budget ────────────────────────────────────────────────────────────────


def test_budget_needs_a_positive_limit():
    with pytest.raises(DojoConfigError, match="at least one"):
        Budget()
    for bad in ({"max_rollouts": 0}, {"max_cost_usd": -1}, {"max_hours": float("inf")}):
        with pytest.raises(DojoConfigError, match="positive"):
            Budget(**bad)
    with pytest.raises(DojoConfigError, match="whole number"):
        Budget(max_rollouts=2.5)


def test_num_rollout_is_the_rollout_limit_or_unbounded():
    assert Budget(max_rollouts=12, max_hours=1).num_rollout == 12
    assert Budget(max_cost_usd=10).num_rollout == budget_mod.UNBOUNDED_ROLLOUTS


def test_driver_env_prices_the_run():
    env = Budget(max_cost_usd=36, max_hours=2).driver_env(
        started_at=100.0, gpus=8, gpu_hour_usd=4.5
    )
    state = json.loads(env[budget_mod.BUDGET_ENV])
    assert state == {
        "started_at": 100.0,
        "max_seconds": 7200,
        "max_cost_usd": 36,
        "usd_per_second": 0.01,
    }
    with pytest.raises(DojoConfigError, match="list price"):
        Budget(max_cost_usd=1).driver_env(started_at=0, gpus=8, gpu_hour_usd=None)
    # Time-only budgets need no price.
    Budget(max_hours=1).driver_env(started_at=0, gpus=8, gpu_hour_usd=None)


def test_exhausted_by_time_or_cost():
    state = {
        "started_at": 0.0,
        "max_seconds": 3600.0,
        "max_cost_usd": 20.0,
        "usd_per_second": 0.01,  # $20 after 2000 s
    }
    assert budget_mod.exhausted(state, 1999) is None
    assert budget_mod.exhausted(state, 2000).startswith("cost: $20.00 of $20")
    assert budget_mod.exhausted({**state, "max_cost_usd": None}, 3600).startswith(
        "time: 1.00 of 1 hours"
    )


def test_budget_final_only_acts_with_a_budget(monkeypatch):
    args = SimpleNamespace(num_rollout=10)
    monkeypatch.delenv(budget_mod.BUDGET_ENV, raising=False)
    assert not budget_mod.budget_final(args, 9)  # unchanged without a budget

    env = Budget(max_hours=1).driver_env(started_at=0.0, gpus=1, gpu_hour_usd=None)
    monkeypatch.setenv(budget_mod.BUDGET_ENV, env[budget_mod.BUDGET_ENV])
    monkeypatch.setattr(budget_mod.time, "time", lambda: 10.0)
    assert not budget_mod.budget_final(args, 3)
    assert budget_mod.budget_final(args, 9)  # the last rollout
    monkeypatch.setattr(budget_mod.time, "time", lambda: 3600.0)
    assert budget_mod.budget_final(args, 3)


def test_recipe_budget_must_agree_with_num_rollout():
    Qwen3_5_4B_Recipe(num_rollout=3, budget=Budget(max_rollouts=3))
    Qwen3_5_4B_Recipe(num_rollout=50, budget=Budget(max_hours=1))
    with pytest.raises(DojoConfigError, match="disagrees"):
        Qwen3_5_4B_Recipe(num_rollout=5, budget=Budget(max_rollouts=3))


def test_budget_is_not_a_framework_flag():
    from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe

    assert "budget" in Qwen3_5_4B_Recipe._SKIP_FIELDS
    assert "budget" in MilesRecipe._SKIP_FIELDS


# ── Driver loop patch ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    ["slime_train", "slime_train_async", "miles_train", "miles_train_async"],
)
def test_patch_applies_to_each_framework_loop(name):
    src = (TESTDATA / f"{name}.py.input").read_text()
    patched = patch_source(src)
    compile(patched, name, "exec")
    assert patch_source(patched) == patched  # idempotent
    assert patched.count("_md_final = _md_budget_final(args, rollout_id)") == 1
    assert "if _md_final or " in patched
    assert f"if _md_final:  # {MARKER}\n" in patched


def test_patch_rejects_an_unrecognized_loop():
    with pytest.raises(RuntimeError, match="no `for rollout_id"):
        patch_source("import os\n\ndef train(args):\n    pass\n")


def _run_patched_slime(monkeypatch, *, num_rollout, budget_env):
    """Run the patched real slime train() with Ray and slime faked out."""
    events = []

    class Actor:
        def update_weights(self):
            pass

        def async_train(self, rollout_id, data):
            events.append(("train", rollout_id))

        def save_model(self, rollout_id, force_sync=False):
            events.append(("save", rollout_id, force_sync))

        def clear_memory(self):
            pass

    class Method:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            events.append((self.name, *args))

    class RolloutManager:
        generate = Method("generate")
        dispose = Method("dispose")
        eval = Method("eval")
        onload_weights = onload_kv = offload = save = Method("noop")

    def should_run_periodic_action(rollout_id, interval, per_epoch=None, total=None):
        if interval is None:
            return False
        return (rollout_id + 1) % interval == 0 or rollout_id == total - 1

    fakes = {
        "ray": SimpleNamespace(get=lambda x: x),
        "slime": types.ModuleType("slime"),
        "slime.ray": types.ModuleType("slime.ray"),
        "slime.ray.placement_group": SimpleNamespace(
            create_placement_groups=lambda args: {"rollout": None},
            create_rollout_manager=lambda args, pg: (RolloutManager(), 1000),
            create_training_models=lambda args, pgs, rm: (Actor(), None),
        ),
        "slime.utils": types.ModuleType("slime.utils"),
        "slime.utils.arguments": SimpleNamespace(parse_args=None),
        "slime.utils.logging_utils": SimpleNamespace(
            configure_logger=lambda: None,
            finish_tracking=lambda args: None,
            init_tracking=lambda args: None,
        ),
        "slime.utils.misc": SimpleNamespace(
            should_run_periodic_action=should_run_periodic_action
        ),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    if budget_env is None:
        monkeypatch.delenv(budget_mod.BUDGET_ENV, raising=False)
    else:
        monkeypatch.setenv(budget_mod.BUDGET_ENV, budget_env)

    src = patch_source((TESTDATA / "slime_train.py.input").read_text())
    namespace = {"__name__": "slime_train"}
    exec(compile(src, "train.py", "exec"), namespace)
    args = SimpleNamespace(
        release_train=False,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        num_rollout=num_rollout,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=True,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
    )
    namespace["train"](args)
    return events


def test_patched_slime_loop_saves_and_stops_when_budget_runs_out(monkeypatch):
    clock = iter([0.0, 0.0, 30.0])  # rollouts 0, 1 in budget; 2 runs it out
    monkeypatch.setattr(budget_mod.time, "time", lambda: next(clock))
    env = Budget(max_hours=0.005).driver_env(started_at=0.0, gpus=1, gpu_hour_usd=None)

    events = _run_patched_slime(
        monkeypatch, num_rollout=1_000_000, budget_env=env[budget_mod.BUDGET_ENV]
    )

    trained = [e[1] for e in events if e[0] == "train"]
    assert trained == [0, 1, 2]
    assert [e for e in events if e[0] == "save"] == [("save", 2, True)]
    assert events[-1][0] == "dispose"


def test_patched_slime_loop_saves_on_its_last_budgeted_rollout(monkeypatch):
    env = Budget(max_rollouts=3).driver_env(started_at=0.0, gpus=1, gpu_hour_usd=None)
    events = _run_patched_slime(
        monkeypatch, num_rollout=3, budget_env=env[budget_mod.BUDGET_ENV]
    )
    assert [e[1] for e in events if e[0] == "train"] == [0, 1, 2]
    assert [e for e in events if e[0] == "save"] == [("save", 2, True)]


def test_patched_slime_loop_is_unchanged_without_a_budget(monkeypatch):
    events = _run_patched_slime(monkeypatch, num_rollout=3, budget_env=None)
    assert [e[1] for e in events if e[0] == "train"] == [0, 1, 2]
    assert [e for e in events if e[0] == "save"] == []  # no save_interval
