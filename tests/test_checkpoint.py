import types
from dataclasses import dataclass

import modal
import pytest

from modal_dojo.common import checkpoint as checkpoint_mod
from modal_dojo.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_megatron_checkpoint_to_hf,
    volume_relative_path,
)
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.framework import Framework
from modal_dojo.common.launcher_helpers import (
    compute_recipe_save_root,
    compute_save_root,
)
from modal_dojo.common.models import ModelConfig, Qwen3_5_4B
from modal_dojo.common.run import TrainingRun, set_checkpoint_location


class _CheckpointVolume:
    def __init__(self, names: list[str]):
        self.names = names

    def read_file(self, path: str):
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name not in self.names:
            raise FileNotFoundError(path)
        return []


def _megatron_checkpoint() -> Checkpoint:
    return Checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_10",
        path="/checkpoints/run/iter_10",
        timestamp=1.0,
        training_run_id="run",
        app_name="app",
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _patch_volume(monkeypatch, volume: _CheckpointVolume) -> None:
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )


def test_convert_megatron_checkpoint_to_hf_returns_hf_checkpoints_unchanged() -> None:
    checkpoint = Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name="iter_0000010_hf",
        path="/checkpoints/run/iter_0000010_hf",
        timestamp=1.0,
        checkpoints_volume_name="gym-checkpoints",
    )

    result = convert_megatron_checkpoint_to_hf(
        checkpoint, ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result is checkpoint


def test_save_root_must_stay_inside_checkpoint_volume() -> None:
    with pytest.raises(DojoConfigError, match="outside Volume mount"):
        compute_save_root(
            "/tmp/checkpoints",
            recipe_default_save_root="/checkpoints",
            mounted_save_root="/checkpoints",
            training_run_id="run-1",
        )


def test_extra_config_save_wins_and_is_pinned_to_scoped_root() -> None:
    recipe = types.SimpleNamespace(
        save="/checkpoints",
        extra_config={"save": "/checkpoints/custom", "qkv_format": "bshd"},
    )

    root = compute_recipe_save_root(
        recipe,
        recipe_default_save_root="/checkpoints",
        mounted_save_root="/checkpoints",
        training_run_id="run-1",
    )

    assert root == "/checkpoints/custom/run-1"
    assert recipe.extra_config == {
        "save": "/checkpoints/custom",
        "qkv_format": "bshd",
    }


def test_two_save_roots_are_siblings_and_leave_recipe_unmutated() -> None:
    recipe = types.SimpleNamespace(
        save="/checkpoints",
        extra_config={"save": "/checkpoints/custom"},
    )

    first = compute_recipe_save_root(
        recipe,
        recipe_default_save_root="/checkpoints",
        mounted_save_root="/checkpoints",
        training_run_id="run-a",
    )
    second = compute_recipe_save_root(
        recipe,
        recipe_default_save_root="/checkpoints",
        mounted_save_root="/checkpoints",
        training_run_id="run-b",
    )

    assert first == "/checkpoints/custom/run-a"
    assert second == "/checkpoints/custom/run-b"
    assert recipe.extra_config == {"save": "/checkpoints/custom"}
    assert recipe.save == "/checkpoints"


def test_recipe_save_is_used_when_extra_config_omits_save() -> None:
    recipe = types.SimpleNamespace(
        save="/checkpoints", extra_config={"qkv_format": "thd"}
    )

    root = compute_recipe_save_root(
        recipe,
        recipe_default_save_root="/checkpoints",
        mounted_save_root="/checkpoints",
        training_run_id="run-1",
    )

    assert root == "/checkpoints/run-1"
    assert recipe.extra_config == {"qkv_format": "thd"}


@pytest.mark.parametrize("training_run_id", ["../outside", "/tmp/outside"])
def test_training_run_id_cannot_escape_checkpoint_volume(
    training_run_id: str,
) -> None:
    with pytest.raises(DojoConfigError, match="outside Volume mount"):
        compute_save_root(
            "/checkpoints",
            recipe_default_save_root="/checkpoints",
            mounted_save_root="/checkpoints",
            training_run_id=training_run_id,
        )


def test_relative_checkpoint_path_is_rejected() -> None:
    with pytest.raises(DojoConfigError, match="must be absolute POSIX paths"):
        volume_relative_path("run/iter_10", "/checkpoints")


def test_convert_reuses_when_marker_present(monkeypatch) -> None:
    _patch_volume(
        monkeypatch,
        _CheckpointVolume(
            [
                "config.json",
                "model.safetensors",
                checkpoint_mod._CONVERT_COMPLETE_MARKER,
            ]
        ),
    )

    result = convert_megatron_checkpoint_to_hf(
        _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result.checkpoint_type is CheckpointType.hf
    assert result.path == "/checkpoints/run/iter_10_hf"
    assert result.name == "iter_10_hf"


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["config.json", "model.safetensors"],
    ],
    ids=["missing", "unmarked"],
)
def test_convert_runs_without_marker(monkeypatch, names: list[str]) -> None:
    _patch_volume(monkeypatch, _CheckpointVolume(names))
    monkeypatch.setattr(
        modal,
        "App",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("convert")),
    )

    with pytest.raises(RuntimeError, match="convert"):
        convert_megatron_checkpoint_to_hf(
            _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
        )


@dataclass
class _DirEntry:
    path: str
    mtime: float = 1.0
    is_directory: bool = False

    def is_dir(self) -> bool:
        return self.is_directory


class _ListingVolume:
    def __init__(
        self,
        tree: dict[str, list[_DirEntry]],
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.tree = tree
        self.files = files or {}
        self.iterdir_calls: list[str] = []

    def iterdir(self, path: str, *, recursive: bool = False):
        del recursive
        key = path.rstrip("/") or "."
        self.iterdir_calls.append(key)
        return list(self.tree.get(key, []))

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]


def _run_with_checkpoint_location() -> TrainingRun:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
        app_name="slime",
        source_model=Qwen3_5_4B(),
    )
    set_checkpoint_location(
        run,
        checkpoint_dir="/checkpoints/run-1",
        checkpoints_volume_name="slime-slime4brecipe-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    return run


def test_run_lists_megatron_iter_at_or_before_tracker(monkeypatch, fake_volume) -> None:
    run = _run_with_checkpoint_location()
    volume = _ListingVolume(
        {
            "run-1": [
                _DirEntry("iter_0000001", is_directory=True),
                _DirEntry("iter_0000002", is_directory=True),
                _DirEntry("iter_0000003", is_directory=True),
                _DirEntry("iter_0000004_hf", is_directory=True),
            ],
        },
        files={"run-1/latest_checkpointed_iteration.txt": b"1\n"},
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    checkpoints = run.checkpoints()
    latest = run.latest_checkpoint()

    assert [checkpoint.name for checkpoint in checkpoints] == ["iter_0000001"]
    assert latest is not None
    assert latest.name == "iter_0000001"
    assert latest.checkpoint_type is CheckpointType.megatron
    assert run.model.model_path == "/checkpoints/run-1/iter_0000001"


def test_run_hides_checkpoints_without_tracker(monkeypatch, fake_volume) -> None:
    run = _run_with_checkpoint_location()
    volume = _ListingVolume(
        {
            "run-1": [_DirEntry("iter_0000001", is_directory=True)],
        }
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    assert run.checkpoints() == []


def test_listing_without_checkpoint_location_is_empty(fake_volume) -> None:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
    )

    assert run.checkpoints() == []


def test_latest_checkpoint_sees_location_written_after_launch(
    monkeypatch, fake_volume
) -> None:
    live = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
    )
    _run_with_checkpoint_location().save()
    volume = _ListingVolume(
        {
            "run-1": [_DirEntry("iter_0000001", is_directory=True)],
        },
        files={"run-1/latest_checkpointed_iteration.txt": b"1\n"},
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    checkpoint = live.latest_checkpoint()

    assert checkpoint is not None
    assert checkpoint.name == "iter_0000001"


def test_list_checkpoints_makes_one_iterdir_regardless_of_n(
    monkeypatch, fake_volume
) -> None:
    run = _run_with_checkpoint_location()
    volume = _ListingVolume(
        {
            "run-1": [
                _DirEntry(f"iter_{i:07d}", is_directory=True) for i in range(1, 6)
            ],
        },
        files={"run-1/latest_checkpointed_iteration.txt": b"5\n"},
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    checkpoints = run.checkpoints()

    assert [checkpoint.name for checkpoint in checkpoints] == [
        f"iter_{i:07d}" for i in range(1, 6)
    ]
    assert volume.iterdir_calls == ["run-1"]
