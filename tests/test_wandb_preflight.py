"""The W&B pre-flight: turn a recurring mid-training wandb CommError into an
early, actionable failure (missing key, or no write access).
"""

import sys
import types
from unittest.mock import Mock

import pytest

from modal_dojo.common.metrics import apply_metric_image
from modal_dojo.common.wandb import WandbConfig, preflight_wandb
from modal_dojo.frameworks.slime.launcher import (
    _preflight_wandb as _slime_preflight_wandb,
)

_CFG = WandbConfig(project="qwen3-asr-rl", modal_wandb_secret_name="wandb-secret")


def _stub_wandb(monkeypatch, **attrs):
    """Stand in for the lazily-imported ``wandb`` module with a fake exposing *attrs*.

    ``preflight_wandb`` does ``import wandb`` inside the function, so preloading a
    fake into ``sys.modules`` intercepts it — no real library, network, or login.
    """
    attrs.setdefault("teardown", Mock())
    module = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_preflight_raises_clear_error_without_key(monkeypatch):
    """No WANDB_API_KEY → a clear error naming the missing var, before any GPU work."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        preflight_wandb(_CFG)


def test_preflight_wraps_access_failure(monkeypatch):
    """Key present but W&B rejects the write → the raw wandb error is
    re-raised as a RuntimeError that names the project."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    def login_without_write_access(**_):
        raise Exception("user does not have models write access")

    module = _stub_wandb(monkeypatch, login=login_without_write_access)
    with pytest.raises(RuntimeError, match="W&B pre-flight failed.*qwen3-asr-rl"):
        preflight_wandb(_CFG)
    module.teardown.assert_called_once_with()


def test_preflight_returns_entity(monkeypatch):
    """Successful preflight returns the W&B entity string."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    class _FakeRun:
        entity = "my-team"
        project = "qwen3-asr-rl"
        id = "abc123"

    class _FakeApi:
        def __init__(self, **_):
            pass

        def run(self, path):
            return types.SimpleNamespace(delete=lambda: None)

    module = _stub_wandb(
        monkeypatch,
        login=lambda **_: None,
        init=lambda **_: _FakeRun(),
        finish=lambda: None,
        Settings=lambda **_: {},
        Api=_FakeApi,
    )
    entity = preflight_wandb(_CFG)
    assert entity == "my-team"
    module.teardown.assert_called_once_with()


def test_slime_preflight_delegates_to_common(monkeypatch):
    """The slime launcher's _preflight_wandb delegates to the common function."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    class _FakeRun:
        entity = "slime-team"
        project = "qwen3-asr-rl"
        id = "xyz789"

    class _FakeApi:
        def __init__(self, **_):
            pass

        def run(self, path):
            return types.SimpleNamespace(delete=lambda: None)

    _stub_wandb(
        monkeypatch,
        login=lambda **_: None,
        init=lambda **_: _FakeRun(),
        finish=lambda: None,
        Settings=lambda **_: {},
        Api=_FakeApi,
    )
    entity = _slime_preflight_wandb(_CFG)
    assert entity == "slime-team"


def test_wandb_config_uses_the_provider_neutral_recipe_field():
    from modal_dojo import MetricConfig, Qwen3_4B_Recipe
    from modal_dojo.common.launcher_helpers import build_app_tags

    with pytest.raises(TypeError, match="abstract"):
        MetricConfig()

    metric = WandbConfig(project="training", group="experiment")
    recipe = Qwen3_4B_Recipe(metrics=metric)

    assert isinstance(metric, MetricConfig)
    assert recipe.metrics is metric
    assert "--use-wandb" in recipe.cli_args()
    assert "label" not in metric.metadata(run_id="run-1")
    tags = build_app_tags(
        framework="slime",
        model=types.SimpleNamespace(model_name="Qwen/Qwen3-4B"),
        recipe_app_tags={},
        metrics=metric,
    )
    assert tags["_modal_metric_project"] == "training"
    assert tags["_modal_metric_group"] == "experiment"
    assert "_modal_wandb_project" not in tags
    assert len(tags) <= 8


@pytest.mark.parametrize(
    "config_key,environment_key",
    [("config-key", ""), ("config-key", "env-key"), ("", "")],
)
def test_wandb_credentials_use_environment_instead_of_cli(
    monkeypatch, config_key, environment_key
):
    from modal_dojo.common.metrics import metric_cli_fields

    monkeypatch.setenv("WANDB_API_KEY", environment_key)
    metric = WandbConfig(project="test", key=config_key)
    assert "wandb_key" not in metric_cli_fields(metric)
    env = metric.runtime_env(run_id="run-1")
    assert env.get("WANDB_API_KEY", "") == (environment_key or config_key)
    assert env["WANDB_RUN_ID"] == "run-1"
    assert metric.key == config_key


class _FakeImage:
    def __init__(self) -> None:
        self.packages: list[str] = []
        self.commands: list[str] = []

    def uv_pip_install(self, package: str) -> "_FakeImage":
        self.packages.append(package)
        return self

    def run_commands(self, command: str) -> "_FakeImage":
        self.commands.append(command)
        return self


def test_wandb_image_pins_generate_id_release():
    image = _FakeImage()
    result = apply_metric_image(image, WandbConfig(project="sf3"))

    assert result is image
    assert image.packages == ["wandb==0.28.1"]
