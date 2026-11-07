"""SpindleRecipe's Modal app: its image, shipped Harbor tasks and run status."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.frameworks.spindle import app
from modal_dojo.frameworks.spindle.reporting import DojoReporter, new_training_run


class RecordingImage:
    def __init__(self) -> None:
        self.dirs: dict[str, Path] = {}

    def add_local_dir(self, local_path, *, remote_path) -> RecordingImage:
        self.dirs[remote_path] = Path(local_path)
        return self


def test_image_installs_modal_dojo_and_spindle_pins_without_markers():
    pins = app.spindle_requirements()
    names = {pin.split("=")[0].split("~")[0] for pin in pins}
    assert {"modal", "httpx", "tinker", "tinker-cookbook"} <= names
    assert not any(";" in pin for pin in pins)


def _harbor_tasks(root: Path, *names: str) -> None:
    for name in names:
        (root / name / "environment").mkdir(parents=True)
        (root / name / "instruction.md").write_text(f"Do {name}")
        (root / name / "task.toml").write_text("")


def test_local_harbor_tasks_ship_to_the_container(tmp_path):
    chz = pytest.importorskip("chz")
    from tinker_cookbook.rl.train import Config

    from modal_dojo import DojoDatasetBuilder
    from modal_dojo.common.dataset import HarborDataset

    _harbor_tasks(tmp_path / "tasks", "fix-git", "regex-log", "skip-me")
    dataset = HarborDataset(
        path=str(tmp_path / "tasks"), task_names=["fix-git", "regex-log"]
    )
    builder = DojoDatasetBuilder.from_dataset(
        dataset,
        model_name="Qwen/Qwen3.5-2B",
        renderer_name="qwen3_5_disable_thinking",
        batch_size=2,
        group_size=2,
    )
    config = Config(
        recipe_name="test",
        learning_rate=1e-5,
        dataset_builder=builder,
        model_name="Qwen/Qwen3.5-2B",
        max_tokens=64,
        log_path=str(tmp_path / "log"),
    )
    config, image = app.ship_harbor_tasks(config, RecordingImage())

    remote = f"{app.HARBOR_TASKS_PATH}/tasks"
    assert image.dirs == {
        f"{remote}/fix-git": tmp_path / "tasks" / "fix-git",
        f"{remote}/regex-log": tmp_path / "tasks" / "regex-log",
    }
    shipped = config.dataset_builder.dataset
    assert (shipped.path, shipped.task_names) == (remote, ["fix-git", "regex-log"])
    assert dataset.path == str(tmp_path / "tasks")
    assert chz.chz_fields(config)


def test_cookbook_harbor_tasks_ship_once_each(tmp_path):
    pytest.importorskip("chz")
    from tinker_cookbook.recipes.harbor_rl.harbor_env import (
        HarborDatasetBuilder,
        HarborTask,
    )
    from tinker_cookbook.rl.train import Config

    _harbor_tasks(tmp_path, "fix-git")
    task = HarborTask(
        task_name="fix-git", instruction="", task_dir=tmp_path / "fix-git"
    )
    config = Config(
        recipe_name="test",
        learning_rate=1e-5,
        dataset_builder=HarborDatasetBuilder(
            tasks=[task, task],
            batch_size=1,
            group_size=1,
            model_name="Qwen/Qwen3.5-2B",
        ),
        model_name="Qwen/Qwen3.5-2B",
        max_tokens=64,
        log_path=str(tmp_path / "log"),
    )
    config, image = app.ship_harbor_tasks(config, RecordingImage())

    remote = f"{app.HARBOR_TASKS_PATH}/0/fix-git"
    assert image.dirs == {remote: tmp_path / "fix-git"}
    assert [t.task_dir for t in config.dataset_builder.tasks] == [Path(remote)] * 2


def test_config_with_inline_reward_pickles_for_the_container(tmp_path):
    pytest.importorskip("chz")
    import cloudpickle
    from tinker_cookbook.rl.train import Config

    from modal_dojo import DojoDatasetBuilder
    from modal_dojo.common.dataset import HuggingFaceDataset

    builder = DojoDatasetBuilder.from_dataset(
        HuggingFaceDataset(
            hf_repo="openai/gsm8k",
            hf_config="main",
            input_column="question",
            output_column="answer",
        ),
        model_name="Qwen/Qwen3.5-2B",
        renderer_name="qwen3_5_disable_thinking",
        batch_size=2,
        group_size=2,
        reward_fn=lambda response, label: float(response == label),
    )
    config = Config(
        recipe_name="test",
        learning_rate=1e-5,
        dataset_builder=builder,
        model_name="Qwen/Qwen3.5-2B",
        max_tokens=64,
        log_path=str(tmp_path),
    )
    restored = cloudpickle.loads(cloudpickle.dumps(config))
    assert restored.dataset_builder.reward_fn("4", "4") == 1.0


class _Client:
    async def aclose(self):
        pass


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (None, TrainingRunStatus.COMPLETED),
        (asyncio.CancelledError(), TrainingRunStatus.STOPPED),
        (KeyboardInterrupt(), TrainingRunStatus.STOPPED),
        (RuntimeError("sampler died"), TrainingRunStatus.FAILED),
    ],
)
def test_the_driver_records_how_the_run_ended(fake_volume, tmp_path, error, status):
    config = SimpleNamespace(
        model_name="Qwen/Qwen3.5-2B",
        dataset_builder=object(),
        log_path=str(tmp_path),
    )
    run = new_training_run(config, "spindle-test")
    reporter = DojoReporter(config, run, dashboard_url="https://dojo", token="t")
    reporter.client = _Client()
    reporter.workers = []
    asyncio.run(reporter.__aexit__(type(error), error, None))

    saved = TrainingRun.from_id("spindle-test")
    assert saved.status is status
    assert saved.metadata["last_attempt_status"] == status.value
    if status is TrainingRunStatus.FAILED:
        assert "sampler died" in saved.error_message
