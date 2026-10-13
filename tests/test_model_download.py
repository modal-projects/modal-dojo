"""``HFModelConfiguration.download()`` against a populated ``model_path``.

A populated ``model_path`` already holds the weights to load. Mirroring the
base snapshot preserves those weights.
"""

from __future__ import annotations

import pytest

from modal_dojo.common.models.base import HFModelConfiguration


class _Model(HFModelConfiguration):
    model_name = "org/base-model"


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    snapshot_dir = tmp_path / "hub" / "snapshot"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.safetensors").write_text("base weights")
    (snapshot_dir / "config.json").write_text("{}")

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot_dir)
    )
    return snapshot_dir


def test_populated_model_path_keeps_its_weights(snapshot, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "r0_hf"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_text("trained weights")

    _Model(model_path=str(checkpoint)).download()

    assert (checkpoint / "model.safetensors").read_text() == "trained weights"
    assert not (checkpoint / "config.json").exists()


def test_empty_model_path_receives_the_snapshot(snapshot, tmp_path):
    destination = tmp_path / "model"

    _Model(model_path=str(destination)).download()

    assert (destination / "model.safetensors").read_text() == "base weights"
    assert (destination / "config.json").read_text() == "{}"


def test_model_path_equal_to_snapshot_dir_is_left_alone(snapshot):
    _Model(model_path=str(snapshot)).download()

    assert (snapshot / "model.safetensors").read_text() == "base weights"


def test_download_pins_the_configured_revision(tmp_path, monkeypatch):
    import huggingface_hub

    seen: dict = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    class _Pinned(_Model):
        revision = "0" * 40

    _Pinned().download()

    assert seen == {"repo_id": "org/base-model", "revision": "0" * 40}


def test_every_bundled_model_pins_a_hub_commit():
    import inspect
    import re

    from modal_dojo.common import models

    for name in models.__all__:
        cls = getattr(models, name)
        if not (inspect.isclass(cls) and issubclass(cls, HFModelConfiguration)):
            continue
        if cls is HFModelConfiguration:
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", cls.revision or ""), name
