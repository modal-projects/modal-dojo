import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_training_gym import common
from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.frameworks.miles import launcher
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def build_app(monkeypatch, model, environment=None, mount="/checkpoints"):
    monkeypatch.setattr(
        launcher,
        "App",
        lambda *a, **k: SimpleNamespace(function=lambda **kw: lambda fn: fn),
    )
    monkeypatch.setattr(common, "hf_secrets", lambda: [])
    monkeypatch.setattr(common, "proxy_auth_secrets", lambda: [])
    monkeypatch.setattr(launcher, "hf_secrets", lambda: [])
    monkeypatch.setattr(launcher, "proxy_auth_secrets", lambda: [])
    monkeypatch.setattr(
        launcher, "_build_miles_base_image", lambda *a: launcher.Image.debian_slim()
    )
    return launcher.build_miles_app(
        training_run_id="test-model-paths",
        miles=MilesRecipe(environment=environment or {}),
        model=model,
        dataset=HuggingFaceDataset(
            "org/data", input_column="prompt", output_column="label"
        ),
        checkpoint=SimpleNamespace(
            checkpoints_volume_name="test-checkpoints", checkpoints_mount_path=mount
        ),
    )


@pytest.mark.parametrize("entrypoint", ["download", "train"])
@pytest.mark.parametrize("source", ["model_path", "hub", "absolute_name"])
def test_remote_code_materializes_the_actual_model(
    monkeypatch, tmp_path, entrypoint, source
):
    blob = tmp_path / "blob.py"
    blob.write_text("value = 1\n")
    local = tmp_path / "model"
    local.mkdir()
    code = local / "model.py"
    code.symlink_to(blob)
    model = ModelConfig(
        model_name=str(local) if source == "absolute_name" else "org/model",
        model_path=str(local) if source == "model_path" else None,
    )
    resolver = Mock(return_value=str(local))
    monkeypatch.setattr(launcher, "resolve_checkpoint_ref", resolver)
    app = build_app(monkeypatch, model)
    scope = inspect.getclosurevars(getattr(app, entrypoint)).nonlocals
    if entrypoint == "download":
        scope = inspect.getclosurevars(scope["download"]).nonlocals
    scope["materialize_model_remote_code"]()
    if source == "absolute_name":
        resolver.assert_not_called()
        assert code.is_symlink()
    else:
        resolver.assert_called_once_with(
            str(local) if source == "model_path" else "org/model"
        )
        assert not code.is_symlink()
        assert code.read_text() == blob.read_text()


@pytest.mark.parametrize("mount", ["/checkpoints", "/saved", "/saved/checkpoints"])
@pytest.mark.parametrize(
    "cache_root",
    ["/checkpoints/.kernel-cache/image", "/custom/cache", "/checkpoints-other"],
)
def test_kernel_cache_mount_applies_to_conversion_and_training(
    monkeypatch, mount, cache_root
):
    environment = {
        "TRITON_CACHE_DIR": f"{cache_root}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/torchinductor",
    }
    app = build_app(
        monkeypatch, ModelConfig(model_name="org/model"), environment, mount
    )
    expected_root = (
        mount.rstrip("/") + "/.kernel-cache/image"
        if cache_root == "/checkpoints/.kernel-cache/image"
        else cache_root
    )
    for fn in (app.convert_checkpoint, app.train):
        resolved = inspect.getclosurevars(fn).nonlocals["environment"]
        assert resolved["TRITON_CACHE_DIR"] == f"{expected_root}/triton"
        assert resolved["TORCHINDUCTOR_CACHE_DIR"] == f"{expected_root}/torchinductor"
    assert environment["TRITON_CACHE_DIR"] == f"{cache_root}/triton"
