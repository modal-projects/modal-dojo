from __future__ import annotations

import pytest

import sys
import threading
import types
from dataclasses import fields
from importlib.util import find_spec
from typing import Any

from modal_training_gym.common.metrics import apply_metric_image
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.trackio import (
    TrackioConfig,
    install_wandb_shim,
    require_trackio_destination,
    resolve_trackio_destination,
)


def test_trackio_config_is_provider_specific_without_provider_or_label_fields():
    from modal_training_gym import TrackioConfig as PublicTrackioConfig

    config = TrackioConfig(
        project="rl",
        group="baseline",
        space_id="modal-labs/training-metrics",
        bucket_id="modal-labs/training-metrics",
    )

    assert config.provider == "trackio"
    assert PublicTrackioConfig is TrackioConfig
    assert {field.name for field in fields(config)}.isdisjoint({"provider", "label"})
    assert config.runtime_env(run_id="run-a2") == {
        "TRAINING_GYM_METRIC_PROVIDER": "trackio",
        "TRAINING_GYM_TRACKIO_RUN_NAME": "run-a2",
        "TRACKIO_SPACE_ID": "modal-labs/training-metrics",
        "TRACKIO_BUCKET_ID": "modal-labs/training-metrics",
    }
    assert config.url() == (
        "https://huggingface.co/spaces/modal-labs/training-metrics?project=rl"
    )
    assert config.metadata(run_id="run-a2")["url"] == (
        "https://huggingface.co/spaces/modal-labs/training-metrics"
        "?project=rl&runs=run-a2"
    )
    assert "label" not in config.metadata(run_id="run-a2")


def test_trackio_dashboard_urls_do_not_expose_credentials():
    config = TrackioConfig(
        server_url="https://user:password@metrics.example.com:8443/path"
        "?write_token=secret#fragment"
    )
    assert config.url(run_id="run-a2") == (
        "https://metrics.example.com:8443/path?project=training-gym&runs=run-a2"
    )

    config = TrackioConfig(server_url="https://user:pw@[2001:db8::1]:8443/path")
    assert config.url(run_id="run-a2") == (
        "https://[2001:db8::1]:8443/path?project=training-gym&runs=run-a2"
    )

    config.project = "rl"
    config.dashboard_url = (
        "https://metrics.example.com/view?project=rl&write_token=secret"
    )
    assert config.url(run_id="run-a2") == (
        "https://metrics.example.com/view?project=rl&runs=run-a2"
    )


def test_deploy_to_modal_returns_a_self_hosted_config(monkeypatch):
    calls = {}

    def fake_deploy(**kwargs):
        calls.update(kwargs)
        return "https://example--training-gym-trackio.modal.run"

    monkeypatch.setattr(
        "modal_training_gym.common.trackio._deploy_modal_dashboard", fake_deploy
    )

    config = TrackioConfig.deploy_to_modal(
        project="rl",
        group="baseline",
        app_name="my-trackio",
    )

    assert calls == {
        "app_name": "my-trackio",
        "volume_name": "my-trackio-data",
        "modal_secret_name": "_my-trackio-write-token",
        "TRACKIO_PACKAGE_VERSION": "0.34.0",
    }
    assert config == TrackioConfig(
        project="rl",
        group="baseline",
        server_url="https://example--training-gym-trackio.modal.run",
        dashboard_url="https://example--training-gym-trackio.modal.run",
        modal_secret_name="_my-trackio-write-token",
    )


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


def test_trackio_image_installs_trackio_and_the_metric_bootstrap():
    image = _FakeImage()
    result = apply_metric_image(image, TrackioConfig())

    assert result is image
    assert image.packages == ["trackio==0.34.0"]
    assert len(image.commands) == 1
    assert "_training_gym_metric_mirror.pth" in image.commands[0]
    assert "TRAINING_GYM_METRIC_PROVIDER" in image.commands[0]


def test_trackio_package_version_is_configurable():
    """A pinned default, but bumpable without waiting on a Training Gym release."""
    image = _FakeImage()
    apply_metric_image(image, TrackioConfig(TRACKIO_PACKAGE_VERSION="0.35.0"))

    assert image.packages == ["trackio==0.35.0"]


def test_trackio_wandb_adapter_covers_the_framework_surface(monkeypatch):
    calls: dict[str, Any] = {}

    class _FakeRun:
        name = "fallback-name"
        config = {"learning_rate": 1e-5}

        def log(self, metrics, step=None):
            # The adapter logs through the run object, not the module, so that
            # threads started after init() reach the same run.
            calls.setdefault("logs", []).append((metrics, step))

    fake_trackio = types.ModuleType("trackio")

    # This is the Trackio 0.34.0 keyword surface used by the adapter.
    def fake_init(
        project: str,
        name: str | None = None,
        group: str | None = None,
        space_id: str | None = None,
        server_url: str | None = None,
        bucket_id: str | None = None,
        config: dict[str, Any] | None = None,
        resume: str = "never",
        embed: bool = True,
    ) -> _FakeRun:
        calls["init"] = {
            "project": project,
            "name": name,
            "group": group,
            "space_id": space_id,
            "server_url": server_url,
            "bucket_id": bucket_id,
            "config": config,
            "resume": resume,
            "embed": embed,
        }
        return _FakeRun()

    def fake_finish(*args: Any, **kwargs: Any) -> None:
        calls["finish"] = (args, kwargs)

    fake_trackio.init = fake_init
    fake_trackio.log = lambda data, step=None: calls.setdefault("logs", []).append(
        (data, step)
    )
    fake_trackio.finish = fake_finish
    fake_trackio.save = lambda path: path
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    for module_name in (
        "wandb",
        "wandb.util",
        "wandb.sdk",
        "wandb.sdk.lib",
        "wandb.sdk.lib.runid",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setenv("TRAINING_GYM_TRACKIO_RUN_NAME", "training-run-a2")
    monkeypatch.setenv("TRACKIO_SPACE_ID", "modal-labs/training-metrics")
    monkeypatch.setenv("TRACKIO_SERVER_URL", "https://metrics.example.com")
    monkeypatch.setenv("TRACKIO_BUCKET_ID", "modal-labs/training-metrics")

    install_wandb_shim()

    import wandb
    from wandb.sdk.lib.runid import generate_id

    assert find_spec("wandb") is wandb.__spec__

    settings = wandb.Settings(mode="shared")
    run = wandb.init(
        project="rl",
        entity="ignored",
        id="framework-id",
        group="baseline",
        name="framework-name",
        config={
            "learning_rate": 1e-5,
            "_is_global_batch_size_explicitly_specified": True,
        },
        settings=settings,
        reinit=True,
        dir="/tmp/wandb",
    )

    assert calls["init"] == {
        "project": "rl",
        "name": "training-run-a2",
        "group": "baseline",
        "config": {"learning_rate": 1e-5},
        "resume": "allow",
        "embed": False,
        "space_id": "modal-labs/training-metrics",
        "server_url": "https://metrics.example.com",
        "bucket_id": "modal-labs/training-metrics",
    }
    assert run.id == "training-run-a2"
    assert wandb.run.id == "training-run-a2"
    assert wandb.config == {"learning_rate": 1e-5}
    assert calls["logs"] == [({}, -1)]
    assert len(generate_id()) == 8
    assert wandb.define_metric("train/*", step_metric="train/step") is None

    wandb.log({"loss": 0.5}, commit=False, sync=True)
    assert wandb.save("metrics.json", base_path="/tmp", policy="now") == (
        "metrics.json"
    )
    wandb.finish(exit_code=0, quiet=True)
    assert calls["logs"] == [({}, -1), ({"loss": 0.5}, None)]
    assert calls["finish"] == ((), {})
    assert wandb.run is None


def _install_shim_with_contextvar_trackio(monkeypatch):
    """Fake trackio whose active run lives in a ContextVar, as the real one does."""
    import contextvars

    current_run = contextvars.ContextVar("current_run", default=None)
    logged: list[tuple[dict[str, Any], int | None]] = []

    class _FakeRun:
        name = "fallback-name"
        config: dict[str, Any] = {}

        def log(self, metrics, step=None):
            logged.append((metrics, step))

    def fake_init(**kwargs):
        run = _FakeRun()
        current_run.set(run)
        return run

    def fake_log(data, step=None):
        run = current_run.get()
        if run is None:
            raise RuntimeError("Call trackio.init() before trackio.log().")
        run.log(data, step=step)

    fake_trackio = types.ModuleType("trackio")
    fake_trackio.init = fake_init
    fake_trackio.log = fake_log
    fake_trackio.finish = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    for module_name in (
        "wandb",
        "wandb.util",
        "wandb.sdk",
        "wandb.sdk.lib",
        "wandb.sdk.lib.runid",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setenv("TRAINING_GYM_TRACKIO_RUN_NAME", "training-run-a2")
    install_wandb_shim()
    return logged


def test_trackio_adapter_logs_from_a_thread_started_after_init(monkeypatch):
    """`wandb.run is not None` must imply `wandb.log()` works, in any thread.

    Slime's SGLang engine-metrics loop is a daemon thread that guards on
    `wandb.run` and then logs. trackio keeps its run in a ContextVar, which a
    thread started after init() cannot see, so routing through the module-level
    trackio.log() raised "Call trackio.init() before trackio.log()" there.
    """
    logged = _install_shim_with_contextvar_trackio(monkeypatch)

    import wandb

    wandb.init(project="agentic-harbor")
    error: list[BaseException] = []

    def worker():
        try:
            assert wandb.run is not None
            wandb.log({"sgl_engine/uptime_sec": 1.0}, step=3)
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert error == []
    assert ({"sgl_engine/uptime_sec": 1.0}, 3) in logged


def test_trackio_adapter_preserves_native_train_and_eval_metric_names(monkeypatch):
    """Slime names its per-eval-dataset series ``eval/<subset>``.

    The adapter is the only thing between ``wandb.log`` and Trackio, so a
    rewritten or collapsed key here silently merges every eval subset into one
    chart. Keys and steps must arrive verbatim.
    """
    logged = _install_shim_with_contextvar_trackio(monkeypatch)

    import wandb

    wandb.init(project="agentic-harbor")
    logged.clear()

    wandb.log({"rollout/average_last_reward": 0.625}, step=0)
    wandb.log({"eval/train-2-smoke": 0.5, "eval/eval-2-smoke": 0.25}, step=1)

    assert logged == [
        ({"rollout/average_last_reward": 0.625}, 0),
        ({"eval/train-2-smoke": 0.5, "eval/eval-2-smoke": 0.25}, 1),
    ]


def _stub_discovery(monkeypatch, *, url="https://trackio.example", secret_exists=True):
    monkeypatch.setattr(
        "modal_training_gym.common.trackio.deployed_trackio_url",
        lambda app_name="training-gym-trackio": url,
    )
    monkeypatch.setattr(
        "modal_training_gym.common.trackio._secret_exists",
        lambda name: secret_exists,
    )


def test_a_custom_token_secret_is_not_guessed_at(monkeypatch):
    """deploy_to_modal() takes modal_secret_name, so the name is a convention.

    Adopting a name that doesn't exist mounts nothing and every metric write
    401s -- the silent loss resolution exists to prevent -- so refuse instead.
    """
    _stub_discovery(monkeypatch, secret_exists=False)

    with pytest.raises(TrainingGymConfigError, match="custom modal_secret_name"):
        resolve_trackio_destination(TrackioConfig(project="rl"))


def test_a_failed_resolution_leaves_the_config_untouched(monkeypatch):
    """A half-resolved config is worse than an unresolved one.

    Writing the URLs before the secret check can fail leaves the config looking
    resolved, so a retry takes the early return, skips discovery, and launches
    with no write token -- metrics 401 silently.
    """
    config = TrackioConfig(project="rl")

    _stub_discovery(monkeypatch, secret_exists=False)
    with pytest.raises(TrainingGymConfigError):
        resolve_trackio_destination(config)

    assert config.server_url == ""
    assert config.dashboard_url == ""
    assert config.modal_secret_name == "huggingface-secret"

    # ... so retrying once the secret exists resolves fully rather than no-oping.
    _stub_discovery(monkeypatch, secret_exists=True)
    resolve_trackio_destination(config)

    assert config.server_url == "https://trackio.example"
    assert config.modal_secret_name == "_training-gym-trackio-write-token"


def test_an_explicit_token_secret_is_kept(monkeypatch):
    _stub_discovery(monkeypatch, secret_exists=False)
    config = TrackioConfig(project="rl", modal_secret_name="_my-trackio-token")

    resolve_trackio_destination(config)

    assert config.server_url == "https://trackio.example"
    assert config.modal_secret_name == "_my-trackio-token"


def test_a_bare_config_resolves_to_the_deployed_server(monkeypatch):
    """A recipe names a project; the workspace's server is discovered at launch.

    This is what makes `metrics=TrackioConfig(project=...)` usable as a recipe
    default -- the preset can't know the server URL.
    """
    _stub_discovery(monkeypatch)
    config = TrackioConfig(project="agentic-harbor")

    resolve_trackio_destination(config)

    assert config.server_url == "https://trackio.example"
    assert config.dashboard_url == "https://trackio.example"
    # Ingestion authenticates with the deployed server's write token.
    assert config.modal_secret_name == "_training-gym-trackio-write-token"


def test_an_explicit_destination_is_left_alone(monkeypatch):
    _stub_discovery(monkeypatch, url="https://discovered.example")
    config = TrackioConfig(project="rl", space_id="modal-labs/metrics")

    resolve_trackio_destination(config)

    assert config.server_url == ""
    assert config.modal_secret_name == "huggingface-secret"


def test_no_destination_and_nothing_deployed_is_refused(monkeypatch):
    """Otherwise metrics go to a container-local DB that dies with the run."""
    monkeypatch.setattr(
        "modal_training_gym.common.trackio.deployed_trackio_url",
        lambda app_name="training-gym-trackio": None,
    )
    with pytest.raises(TrainingGymConfigError, match="no destination"):
        resolve_trackio_destination(TrackioConfig(project="rl"))

    # The in-container assertion still holds for anything that slips through.
    with pytest.raises(TrainingGymConfigError, match="no destination"):
        require_trackio_destination(TrackioConfig(project="rl"))
    require_trackio_destination(TrackioConfig(project="rl", server_url="https://x"))
