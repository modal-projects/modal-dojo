"""Local checks for CustomDeployment.launch(unauthenticated=...)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
from modal_dojo import CustomDeployment
from modal_dojo.common.models.base import ModelConfig
from modal_dojo.deploy_recipes.sglang_recipe import SglangRecipe
from modal_dojo.deploy_recipes.sglang_recipe.serve_sglang import (
    build_sglang_serve_app,
)
from modal_dojo.deploy_recipes.vllm_recipe import VllmRecipe
from modal_dojo.deploy_recipes.vllm_recipe.serve_vllm import (
    build_vllm_serve_app,
)


def test_sglang_builder_accepts_unauthenticated() -> None:
    assert "unauthenticated" in inspect.signature(build_sglang_serve_app).parameters


def test_vllm_builder_accepts_unauthenticated() -> None:
    parameters = inspect.signature(build_vllm_serve_app).parameters
    assert parameters["unauthenticated"].default is True


def test_vllm_builder_forwards_unauthenticated_to_app_server() -> None:
    captured: dict = {}

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            self.registered_functions = {}
            self.registered_classes = {}

        def server(self, **kwargs):
            captured.update(kwargs)
            return lambda cls: cls

    with (
        patch("modal.App", FakeApp),
        patch("modal.Image"),
        patch("modal.Volume"),
        patch("modal_dojo.common.hf_secrets", return_value=[]),
    ):
        build_vllm_serve_app(
            recipe=VllmRecipe(),
            app_name="test-serve",
            model_path="test/model",
            served_model_name="model",
            unauthenticated=False,
        )

    assert captured["unauthenticated"] is False


def test_default_unauthenticated_is_true() -> None:
    parameters = inspect.signature(CustomDeployment.launch).parameters
    assert parameters["recipe"].default is None
    assert parameters["unauthenticated"].default is True
    assert {
        "endpoint_name",
        "environment",
        "routing_region",
        "wait_timeout_sec",
    }.isdisjoint(parameters)


def test_sglang_serve_forwards_unauthenticated() -> None:
    captured: dict = {}

    fake_app = MagicMock()
    fake_app.app_id = "ap-test"
    fake_endpoint = MagicMock()
    fake_endpoint.get_url = MagicMock(return_value="https://example.modal.run")
    fake_app.SGLangEndpoint = fake_endpoint

    def _capture_build(**kwargs):
        captured.update(kwargs)
        return fake_app

    with (
        patch(
            "modal_dojo.deploy_recipes.sglang_recipe.serve_sglang.build_sglang_serve_app",
            side_effect=_capture_build,
        ),
        patch(
            "modal_dojo.common.deployment._run_coro",
            return_value="https://example.modal.run",
        ),
    ):
        deployment = CustomDeployment.launch(
            "test/model",
            recipe=SglangRecipe(),
            app_name="custom-app",
            served_model_name="custom-served-model",
            unauthenticated=True,
        )

    assert captured.get("unauthenticated") is True
    assert captured["app_name"] == "custom-app"
    assert captured["model_path"] == "test/model"
    assert captured["served_model_name"] == "custom-served-model"
    assert deployment.model.model_name == "test/model"


def _serve_vllm(*, unauthenticated: bool = True) -> tuple[object, MagicMock]:
    fake_app = MagicMock()
    fake_app.app_id = "ap-test"
    fake_server = MagicMock()
    fake_server.get_url = MagicMock(return_value="https://example.modal.run")
    fake_app.Server = fake_server
    with (
        patch(
            "modal_dojo.deploy_recipes.vllm_recipe.serve_vllm.build_vllm_serve_app",
            return_value=fake_app,
        ) as mock_build,
        patch(
            "modal_dojo.common.deployment._run_coro",
            return_value="https://example.modal.run",
        ),
        patch(
            "modal_dojo.common.deployment.modal_app_dashboard_url",
            return_value="https://modal.com/apps/ap-test",
        ),
    ):
        deployment = CustomDeployment.launch(
            ModelConfig(model_name="test/model"),
            recipe=VllmRecipe(),
            unauthenticated=unauthenticated,
        )
    return deployment, mock_build


@pytest.mark.parametrize("unauthenticated", [True, False])
def test_vllm_serve_forwards_unauthenticated(unauthenticated: bool) -> None:
    deployment, mock_build = _serve_vllm(unauthenticated=unauthenticated)
    assert deployment.url == "https://example.modal.run"
    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["unauthenticated"] is unauthenticated


def test_vllm_serve_forwards_default_unauthenticated() -> None:
    deployment, mock_build = _serve_vllm()
    assert deployment.url == "https://example.modal.run"
    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["unauthenticated"] is True


def test_from_config_missing_unauthenticated_defaults_true() -> None:
    deployment = CustomDeployment.model_validate(
        {
            "deployment_id": "dep-1",
            "url": "https://example.modal.run",
            "deployment_config": {
                "model": {"model_name": "test/model"},
                "app_name": "test-serve",
                "served_model_name": "model",
            },
        }
    )
    assert deployment.unauthenticated is True
    assert deployment.model.model_name == "test/model"
    assert deployment.app_name == "test-serve"
    assert deployment.served_model_name == "model"
