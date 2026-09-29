"""Long setup calls must finish successfully before training can start."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_training_gym import (
    GLM_5_3_Flash,
    GLM_5_3_Flash_Recipe,
    HuggingFaceDataset,
    TrainConfig,
)
from modal_training_gym.common.errors import TrainingGymError
from modal_training_gym.common.run import TrainingRun


@pytest.mark.parametrize("failure", [None, "download", "convert"])
@pytest.mark.parametrize("cached", [False, True])
def test_durable_setup_orders_work_and_stops_on_failure(monkeypatch, failure, cached):
    events = []

    def setup_function(name):
        def get():
            events.append(name)
            if failure == name:
                raise RuntimeError(f"{name} failed")

        return SimpleNamespace(spawn=Mock(return_value=SimpleNamespace(get=get)))

    training_call = SimpleNamespace(object_id="fc-training")

    def spawn_training(**kwargs):
        events.append("train")
        return training_call

    app = SimpleNamespace(
        name="test-setup",
        app_id="ap-test",
        run=lambda **kwargs: nullcontext(),
        download=setup_function("download"),
        resolve_checkpoint=SimpleNamespace(
            remote=Mock(return_value=None if cached else "/hf/model")
        ),
        convert_checkpoint=setup_function("convert"),
        train=SimpleNamespace(spawn=Mock(side_effect=spawn_training)),
    )
    monkeypatch.setattr(TrainConfig, "_build_app", lambda *args: app)
    monkeypatch.setattr(TrainConfig, "_build_status_display", lambda *args: Mock())
    monkeypatch.setattr(TrainingRun, "save", lambda *args: None)
    for target in (
        "modal_training_gym.cli.setup.ensure_dashboard_deployed",
        "modal_training_gym.common.train.vol_put",
        "modal_training_gym.common.train.maybe_warn_gpu_oom",
        "modal_training_gym.common.status_reporter.enqueue_framework_status",
    ):
        monkeypatch.setattr(target, Mock())
    monkeypatch.setattr(
        "modal_training_gym.common.config.get_framework_status_url", lambda: ""
    )
    stop = Mock()
    monkeypatch.setattr("modal_training_gym.common.modal_lifecycle.stop_app", stop)
    config = TrainConfig(
        model=GLM_5_3_Flash(),
        recipe=GLM_5_3_Flash_Recipe(num_rollout=2),
        dataset=HuggingFaceDataset(
            "test/dataset", input_column="prompt", output_column="label"
        ),
    )

    failed = failure == "download" or (failure == "convert" and not cached)
    if failed:
        with pytest.raises(TrainingGymError, match="before training could begin"):
            config.launch(show_output=False)
        assert "train" not in events
        stop.assert_called_once_with("ap-test")
    else:
        run = config.launch(show_output=False)
        assert run.function_call_id == "fc-training"
        assert events == (
            ["download", "train"] if cached else ["download", "convert", "train"]
        )
        stop.assert_not_called()
    app.download.spawn.assert_called_once()
    if cached or failure == "download":
        app.convert_checkpoint.spawn.assert_not_called()
    else:
        assert app.convert_checkpoint.spawn.call_args.kwargs["hf_path"] == "/hf/model"
