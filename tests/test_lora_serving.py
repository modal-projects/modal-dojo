"""Serving Miles native-LoRA adapter checkpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
from modal_dojo.common.deployment import CustomDeployment
from modal_dojo.common.endpoint import Endpoint
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.framework import Framework
from modal_dojo.common.models.base import ModelConfig
from modal_dojo.common.run import TrainingRun
from modal_dojo.deploy_recipes.sglang_recipe import SglangRecipe
from modal_dojo.deploy_recipes.sglang_recipe.serve_sglang import (
    build_sglang_serve_app,
)
from modal_dojo.deploy_recipes.vllm_recipe import VllmRecipe

_BASE = "deepseek-ai/DeepSeek-V4.1-Flash"


def _adapter() -> Checkpoint:
    return Checkpoint(
        checkpoint_type=CheckpointType.lora_adapter,
        name="iter_0000010",
        path="/checkpoints/run-1/iter_0000010/adapter",
        timestamp=1.0,
        training_run_id="run-1",
        checkpoints_volume_name="miles-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _launch_adapter(recipe=None) -> tuple[CustomDeployment, dict, MagicMock]:
    captured: dict = {}
    fake_app = MagicMock()
    fake_app.app_id = "ap-test"

    def _capture_build(**kwargs):
        captured.update(kwargs)
        return fake_app

    with (
        patch(
            "modal_dojo.deploy_recipes.sglang_recipe.serve_sglang.build_sglang_serve_app",
            side_effect=_capture_build,
        ),
        patch(
            "modal_dojo.common.deployment.convert_native_lora_checkpoint_to_peft",
            return_value="/checkpoints/run-1/iter_0000010/adapter_hf",
        ) as convert,
        patch(
            "modal_dojo.common.deployment._run_coro",
            return_value="https://example.modal.run",
        ),
    ):
        deployment = CustomDeployment.launch(
            ModelConfig(model_name=_BASE),
            _adapter(),
            recipe=recipe or SglangRecipe(),
            served_model_name="dsv41",
        )
    return deployment, captured, convert


def test_adapter_is_served_on_the_base_model() -> None:
    deployment, captured, convert = _launch_adapter()

    convert.assert_called_once()
    assert convert.call_args.args[0] == "/checkpoints/run-1/iter_0000010/adapter"
    assert captured["model_path"] == _BASE
    assert captured["lora_name"] == "iter_0000010"
    assert captured["lora_path"] == "/checkpoints/run-1/iter_0000010/adapter_hf"
    assert captured["checkpoints_volume"] == "miles-checkpoints"
    assert deployment.request_model_name == "dsv41:iter_0000010"


def test_chat_targets_the_adapter(monkeypatch) -> None:
    from modal_dojo.common import deployment as deployment_module

    body: dict = {}

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "OK"}}]}

    def post(url, *, json, timeout, headers):
        body.update(json)
        return _Response()

    import requests

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(deployment_module, "_modal_proxy_auth_headers", dict)
    deployment = CustomDeployment.model_construct(
        deployment_id="test",
        served_model_name="dsv41",
        lora_name="iter_0000010",
        url="https://example.test",
    )

    deployment.generate("hi")

    assert body["model"] == "dsv41:iter_0000010"


def test_vllm_rejects_adapters() -> None:
    with pytest.raises(DojoConfigError, match="SglangRecipe"):
        CustomDeployment.launch(
            ModelConfig(model_name=_BASE), _adapter(), recipe=VllmRecipe()
        )


def test_endpoint_rejects_adapters() -> None:
    with pytest.raises(DojoConfigError, match="LoRA adapter"):
        Endpoint.launch(ModelConfig(model_name=_BASE), _adapter())


def test_sglang_app_loads_adapter_and_warms_it_up() -> None:
    captured: dict = {}

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def server(self, **_kwargs):
            return lambda cls: cls

    def build_server_cmd(**kwargs):
        captured["server_args"] = kwargs["extra_server_args"]
        return []

    def warmup(**kwargs):
        captured["warmup_model"] = kwargs["payload"]["model"]

    endpoint_module = "modal_dojo.deploy_recipes.sglang_recipe._sglang_endpoint"
    with (
        patch("modal.App", FakeApp),
        patch("modal.Image"),
        patch("modal.Volume"),
        patch("modal.enter", lambda: lambda fn: fn),
        patch("modal.exit", lambda: lambda fn: fn),
        patch("modal_dojo.common.hf_secrets", return_value=[]),
        patch(f"{endpoint_module}.build_server_cmd", build_server_cmd),
        patch(f"{endpoint_module}.start_server"),
        patch(f"{endpoint_module}.wait_for_server_ready"),
        patch(f"{endpoint_module}.warmup_chat_completions", warmup),
    ):
        app = build_sglang_serve_app(
            recipe=SglangRecipe(extra_server_args={"--lora-backend": "triton"}),
            app_name="test-serve",
            model_path=_BASE,
            served_model_name="dsv41",
            lora_name="iter_0000010",
            lora_path="/checkpoints/run-1/iter_0000010/adapter_hf",
        )
        app.SGLangEndpoint().start()

    args = captured["server_args"]
    assert args["--enable-lora"] == ""
    assert args["--lora-paths"] == (
        "iter_0000010=/checkpoints/run-1/iter_0000010/adapter_hf"
    )
    assert args["--lora-backend"] == "triton"
    assert args["--served-model-name"] == "dsv41"
    assert captured["warmup_model"] == "dsv41:iter_0000010"


def test_run_serve_uses_base_model_and_training_lora_backend(monkeypatch) -> None:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.MILES,
        config={"recipe": {"sglang_lora_backend": "triton"}},
        source_model={"model_name": _BASE, "model_path": None},
    )
    monkeypatch.setattr(TrainingRun, "latest_checkpoint", lambda self: _adapter())
    launch = MagicMock()
    monkeypatch.setattr(CustomDeployment, "launch", launch)

    run.serve(recipe=SglangRecipe(tp=8))

    model, checkpoint = launch.call_args.args
    recipe = launch.call_args.kwargs["recipe"]
    assert model.model_name == _BASE
    assert checkpoint.checkpoint_type is CheckpointType.lora_adapter
    assert recipe.tp == 8
    assert recipe.extra_server_args == {"--lora-backend": "triton"}


def test_run_serve_without_checkpoint_raises(monkeypatch) -> None:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.MILES,
        config={},
        source_model={"model_name": _BASE},
    )
    monkeypatch.setattr(TrainingRun, "latest_checkpoint", lambda self: None)

    with pytest.raises(ValueError, match="no complete checkpoint"):
        run.serve()
