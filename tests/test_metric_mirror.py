"""Container-side metric mirror: flattening, step semantics, batching, shims."""

from __future__ import annotations

import sys
import types

import pytest

from modal_training_gym.common import metric_mirror, reporting
from modal_training_gym.common.metric_mirror import (
    DashboardMetricConfig,
    MetricMirror,
    flatten_numeric,
    install_wandb_shim,
    patch_wandb_module,
)
from modal_training_gym.common.metrics import (
    apply_metric_image,
    metric_runtime_env,
    metric_secrets,
    preflight_metric,
)
from modal_training_gym.common.trackio import TrackioConfig
from modal_training_gym.common.wandb import WandbConfig


class _Tensor:
    def __init__(self, value, shape=()):
        self._value, self.shape = value, shape

    def item(self):
        if self.shape:
            raise RuntimeError("only one element tensors can be converted")
        return self._value


def test_flatten_keeps_finite_numbers_and_drops_everything_else():
    flat = flatten_numeric(
        {
            "train": {"loss": 0.5, "lr": 1e-4, "nested": {"deep": 2}},
            "reward": 1,
            "tensor": _Tensor(3.5),
            "vector": _Tensor(1.0, shape=(4,)),
            "flag": True,
            "name": "qwen",
            "nan": float("nan"),
            "inf": float("inf"),
            "none": None,
            "image": object(),
        }
    )
    assert flat == {
        "train/loss": 0.5,
        "train/lr": 1e-4,
        "train/nested/deep": 2.0,
        "reward": 1.0,
        "tensor": 3.5,
    }


@pytest.fixture
def sent(monkeypatch) -> list[dict]:
    """Capture what the mirror hands to the reporting queue and keep timers off."""
    batches: list[dict] = []
    monkeypatch.setattr(
        reporting,
        "_enqueue_metric_points",
        lambda payload, final=False: batches.append({**payload, "final": final}),
    )
    monkeypatch.setattr(metric_mirror, "FLUSH_INTERVAL_SECONDS", 3600)
    monkeypatch.setattr(metric_mirror.time, "time", lambda: 1000.0)
    return batches


def test_implicit_step_and_commit_follow_wandb_semantics(sent):
    mirror = MetricMirror("run-1")
    mirror.log({"a": 1})
    mirror.log({"b": 2}, commit=False)
    mirror.log({"c": 3})
    mirror.log({"d": 4}, step=10)
    mirror.log({"e": 5})
    mirror.log({"f": 6}, step=3)  # explicit older step still lands
    mirror.log({"g": 7})
    mirror.log({"loss": 1.0}, step=3)
    mirror.log({"loss": 0.5}, step=3)  # last write wins within a step
    mirror.log({"bad": 1}, step=-1)
    mirror.log({"img": object()}, step=2)
    mirror.flush()
    assert sent == [
        {
            "training_run_id": "run-1",
            "final": False,
            "points": [
                {"step": 0, "metrics": {"a": 1.0}, "time": 1000.0},
                {"step": 1, "metrics": {"b": 2.0, "c": 3.0}, "time": 1000.0},
                {"step": 3, "metrics": {"f": 6.0, "loss": 0.5}, "time": 1000.0},
                {"step": 10, "metrics": {"d": 4.0, "e": 5.0}, "time": 1000.0},
                {"step": 11, "metrics": {"g": 7.0}, "time": 1000.0},
            ],
        }
    ]
    mirror.flush(final=True)
    assert len(sent) == 1  # nothing pending, nothing sent


def test_log_schedules_one_timer_per_batch(sent, monkeypatch):
    monkeypatch.setattr(metric_mirror, "FLUSH_INTERVAL_SECONDS", 0.05)
    mirror = MetricMirror("run-1")
    mirror.log({"a": 1}, step=0)
    mirror.log({"a": 1}, step=1)
    timer = mirror._timer
    assert timer is not None
    timer.join(2)
    assert [p["step"] for p in sent[0]["points"]] == [0, 1]
    assert mirror._timer is None


def test_mirror_log_is_best_effort(sent, monkeypatch):
    monkeypatch.setattr(metric_mirror, "_MIRROR", None)
    monkeypatch.delenv("TRAINING_GYM_TRAINING_RUN_ID", raising=False)
    metric_mirror.mirror_log({"a": 1})  # no run: silently ignored

    monkeypatch.setenv("TRAINING_GYM_TRAINING_RUN_ID", "run-env")
    hooks = []
    monkeypatch.setattr(reporting, "register_pre_drain_hook", hooks.append)
    metric_mirror.mirror_log({"a": 1}, step=2)
    metric_mirror.mirror_log("not a mapping")
    mirror = metric_mirror._MIRROR
    assert mirror is not None and mirror.training_run_id == "run-env"
    assert len(hooks) == 1
    hooks[0]()  # the pre-drain hook is the final flush
    assert sent == [
        {
            "training_run_id": "run-env",
            "final": True,
            "points": [{"step": 2, "metrics": {"a": 1.0}, "time": 1000.0}],
        }
    ]
    monkeypatch.setattr(metric_mirror, "_MIRROR", None)


def test_enqueue_metric_points_derives_url_and_retries(monkeypatch):
    monkeypatch.setenv(
        "TRAINING_GYM_FRAMEWORK_STATUS_URL", "https://dash.test/api/framework-status"
    )
    items, blocking = [], []

    def put(item, block=True, timeout=None):
        items.append(item)
        blocking.append(block)

    monkeypatch.setattr(reporting._REPORT_QUEUE, "put", put)
    monkeypatch.setattr(reporting, "_ensure_worker", lambda **kw: None)
    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", False)
    reporting._enqueue_metric_points({"training_run_id": "r", "points": []})
    reporting._enqueue_metric_points({"training_run_id": "r", "points": []}, final=True)
    assert {item["_url"] for item in items} == {"https://dash.test/api/metric-points"}
    assert [item["_retry_count"] for item in items] == [1, 3]
    assert [item["final"] for item in items] == [False, True]
    assert blocking == [False, True]  # the exit flush waits for queue room
    assert [reporting._is_final_timing(item) for item in items] == [False, True]

    monkeypatch.setattr(reporting, "_REPORTER_DRAINING", True)
    reporting._enqueue_metric_points({"training_run_id": "r", "points": []})
    assert len(items) == 2  # non-final batches are dropped while draining
    reporting._enqueue_metric_points({"training_run_id": "r"}, final=True)
    assert len(items) == 3


def test_drain_compaction_folds_metric_batches_into_one_prioritized_post():
    metrics_url = "https://dash.test/api/metric-points"
    timing_url = "https://dash.test/api/timing-events"
    queue = reporting._REPORT_QUEUE
    while not queue.empty():
        queue.get_nowait()
    for item in (
        {"_url": "https://dash.test/api/training-rollouts", "n": 1},
        {
            "_url": metrics_url,
            "training_run_id": "r",
            "points": [{"step": 10, "metrics": {"loss": 0.8, "lr": 1.0}, "time": 1.0}],
            "final": False,
            "_retry_count": 1,
        },
        {"_url": timing_url, "final": False},
        {"_url": "https://dash.test/api/framework-status", "n": 3},
        {"_url": timing_url, "final": True, "n": 4},
        {
            "_url": metrics_url,
            "training_run_id": "r",
            "points": [{"step": 10, "metrics": {"loss": 0.7}, "time": 2.0}],
            "final": True,
            "_retry_count": 3,
        },
    ):
        queue.put_nowait(item)
    queue.unfinished_tasks = 6
    reporting._compact_report_queue()
    final_timing, metrics, status, rollouts = queue.queue
    assert final_timing["n"] == 4 and status["n"] == 3 and rollouts["n"] == 1
    assert metrics["points"] == [
        {"step": 10, "metrics": {"loss": 0.7, "lr": 1.0}, "time": 2.0}
    ]
    assert metrics["final"] is True and metrics["_retry_count"] == 3
    assert queue.unfinished_tasks == 4
    while not queue.empty():
        queue.get_nowait()


def test_drain_compaction_splits_merged_metrics_at_the_batch_limit(monkeypatch):
    monkeypatch.setattr(reporting, "MAX_METRIC_POINTS_PER_BATCH", 2)
    queue = reporting._REPORT_QUEUE
    while not queue.empty():
        queue.get_nowait()
    for steps in ((0, 1, 2), (3, 4)):
        queue.put_nowait(
            {
                "_url": "https://dash.test/api/metric-points",
                "training_run_id": "r",
                "points": [{"step": s, "metrics": {"a": 1.0}} for s in steps],
                "final": steps == (3, 4),
                "_retry_count": 1,
            }
        )
    reporting._compact_report_queue()
    batches = list(queue.queue)
    assert [[p["step"] for p in b["points"]] for b in batches] == [[0, 1], [2, 3], [4]]
    assert all(b["final"] for b in batches)  # every chunk keeps drain retries
    while not queue.empty():
        queue.get_nowait()


# ── config wiring ─────


class _FakeImage:
    def __init__(self) -> None:
        self.packages: list[str] = []
        self.commands: list[str] = []

    def uv_pip_install(self, package: str) -> _FakeImage:
        self.packages.append(package)
        return self

    def run_commands(self, command: str) -> _FakeImage:
        self.commands.append(command)
        return self


def test_dashboard_config_needs_no_secrets_or_preflight():
    config = DashboardMetricConfig(project="p", group="g", exp_name="e")
    assert metric_runtime_env(config, run_id="r") == {
        "TRAINING_GYM_METRIC_PROVIDER": "dashboard"
    }
    assert config.metadata(run_id="r") == {
        "provider": "dashboard",
        "project": "p",
        "group": "g",
        "entity": "",
        "run_id": "r",
    }
    assert metric_secrets(config) == []
    assert preflight_metric(config) == ""


def test_recipes_default_to_the_dashboard_and_none_opts_out():
    from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe
    from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe

    for recipe_cls in (SlimeRecipe, MilesRecipe):
        assert isinstance(recipe_cls().metrics, DashboardMetricConfig)
        assert "--use-wandb" in recipe_cls().cli_args()
        assert "--use-wandb" not in recipe_cls(metrics=None).cli_args()


@pytest.mark.parametrize(
    "config", [DashboardMetricConfig(), WandbConfig(project="p"), TrackioConfig()]
)
def test_every_provider_installs_the_mirror_pth(config):
    image = apply_metric_image(_FakeImage(), config)
    assert len(image.commands) == 1
    assert "_training_gym_metric_mirror.pth" in image.commands[0]
    assert "TRAINING_GYM_METRIC_PROVIDER" in image.commands[0]
    if isinstance(config, TrackioConfig):
        expected = [f"trackio=={config.TRACKIO_PACKAGE_VERSION}"]
    elif isinstance(config, WandbConfig):
        expected = ["wandb==0.28.1"]
    else:
        expected = []
    assert image.packages == expected
    assert apply_metric_image(_FakeImage(), None).commands == []


def test_bootstrap_dispatches_on_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(
        metric_mirror, "install_wandb_shim", lambda: calls.append("dashboard")
    )
    monkeypatch.setattr(
        metric_mirror, "install_wandb_tee", lambda: calls.append("wandb")
    )
    for provider in ("dashboard", "wandb", "other"):
        monkeypatch.setenv("TRAINING_GYM_METRIC_PROVIDER", provider)
        metric_mirror.bootstrap()
    assert calls == ["dashboard", "wandb"]


# ── wandb surfaces ─────


@pytest.fixture
def isolated_wandb(monkeypatch, sent):
    for name in [n for n in sys.modules if n.split(".")[0] == "wandb"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(metric_mirror, "_MIRROR", MetricMirror("run-shim"))
    yield
    for name in [n for n in sys.modules if n.split(".")[0] == "wandb"]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_dashboard_shim_routes_wandb_calls_to_the_mirror(isolated_wandb, sent):
    install_wandb_shim()
    import wandb  # noqa: PLC0415

    run = wandb.init(project="p", name="exp", config={"lr": 0.1})
    assert wandb.run is run and run.name == "exp" and run.config.lr == 0.1
    run.config.update({"steps": 3})
    assert wandb.config["steps"] == 3
    assert wandb.util.generate_id() and wandb.sdk.lib.runid.generate_id()
    assert wandb.Image("x") is not None
    with pytest.raises(AttributeError):
        wandb.not_a_thing  # noqa: B018

    wandb.log({"train/loss": 0.9, "img": wandb.Image("x")}, step=1)
    wandb.log({"train/loss": 0.4}, step=2)
    run.log({"reward": 1.5}, step=2)
    wandb.define_metric("train/loss", summary="min")
    wandb.save("file.txt")
    run.watch(None)
    assert wandb.login() is True
    wandb.finish()
    assert wandb.run is None
    metric_mirror._MIRROR.flush()
    assert sent[0]["points"] == [
        {"step": 1, "metrics": {"train/loss": 0.9}, "time": 1000.0},
        {"step": 2, "metrics": {"train/loss": 0.4, "reward": 1.5}, "time": 1000.0},
    ]


def _fake_wandb_package() -> types.ModuleType:
    wandb = types.ModuleType("wandb")
    sdk = types.ModuleType("wandb.sdk")
    wandb_run = types.ModuleType("wandb.sdk.wandb_run")

    class Run:
        def __init__(self) -> None:
            self.logged: list[tuple] = []

        def log(self, data, step=None, commit=None, sync=None):
            self.logged.append((dict(data), step, commit))

    wandb_run.Run = Run  # type: ignore[attr-defined]
    sdk.wandb_run = wandb_run  # type: ignore[attr-defined]
    wandb.sdk = sdk  # type: ignore[attr-defined]
    sys.modules.update(
        {"wandb": wandb, "wandb.sdk": sdk, "wandb.sdk.wandb_run": wandb_run}
    )
    return wandb


def test_tee_patches_run_log_and_keeps_calling_wandb(isolated_wandb, sent):
    wandb = _fake_wandb_package()
    patch_wandb_module()
    run = wandb.sdk.wandb_run.Run()
    run.log({"a": 1})
    run.log({"b": 2}, commit=False)
    run.log({"c": 3})
    run.log({"d": 4, "e": "text"}, step=7)
    assert len(run.logged) == 4
    metric_mirror._MIRROR.flush()
    assert sent[0]["points"] == [
        {"step": 0, "metrics": {"a": 1.0}, "time": 1000.0},
        {"step": 1, "metrics": {"b": 2.0, "c": 3.0}, "time": 1000.0},
        {"step": 7, "metrics": {"d": 4.0}, "time": 1000.0},
    ]


def test_trackio_shim_mirrors(isolated_wandb, sent, monkeypatch):
    from modal_training_gym.common import trackio as trackio_module

    logged = []
    fake_trackio = types.SimpleNamespace(
        init=lambda **kwargs: types.SimpleNamespace(
            log=lambda metrics, step=None: logged.append((metrics, step))
        ),
        log=lambda data, step=None: logged.append((data, step)),
        finish=lambda: None,
    )
    monkeypatch.setattr(trackio_module, "import_module", lambda name: fake_trackio)
    trackio_module.install_wandb_shim()
    import wandb  # noqa: PLC0415

    wandb.log({"x": 1.0}, step=4)
    assert logged == [({"x": 1.0}, 4)]
    metric_mirror._MIRROR.flush()
    assert sent[0]["points"] == [{"step": 4, "metrics": {"x": 1.0}, "time": 1000.0}]
