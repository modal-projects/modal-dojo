"""SpindleRecipe's preflight against a running Spindle server."""

from __future__ import annotations

import functools
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from modal_dojo.common.errors import DojoConfigError
from modal_dojo.frameworks.spindle import launcher

URL = "https://spindle.example"
DEPLOYMENTS = [
    {
        "base_model": "Qwen/Qwen3.5-2B",
        "parameterization": "lora",
        "max_context_length": 32768,
    },
    {
        "base_model": "Qwen/Qwen3.5-9B",
        "parameterization": "lora",
        "max_context_length": 16384,
    },
    {
        "base_model": "Qwen/Qwen3.5-9B",
        "parameterization": "lora",
        "max_context_length": 131072,
    },
    {
        "base_model": "Qwen/Qwen3.5-4B",
        "parameterization": "full",
        "max_context_length": 65536,
    },
]


@pytest.fixture
def requests(monkeypatch):
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.headers.get("x-api-key") != "tml-test":
            return httpx.Response(401, json={"error": "unauthorized"})
        if request.url.path == "/api/v1/get_server_capabilities":
            return httpx.Response(
                200, json={"spindle_version": "0.1.4", "spindle_commit": "abc123"}
            )
        return httpx.Response(200, json={"deployments": DEPLOYMENTS})

    client = functools.partial(httpx.Client, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        launcher, "httpx", SimpleNamespace(**{**vars(httpx), "Client": client})
    )
    monkeypatch.setattr(
        launcher, "modal_proxy_auth_headers", lambda: {"Modal-Key": "wk-test"}
    )
    return seen


def check(model_name, max_tokens=64, api_key="tml-test"):
    config = SimpleNamespace(model_name=model_name, max_tokens=max_tokens)
    return launcher.check_server(URL, api_key, config)


def test_records_server_identity_with_proxy_auth(requests):
    assert check("Qwen/Qwen3.5-2B") == {
        "spindle_base_url": URL,
        "spindle_version": "0.1.4",
        "spindle_commit": "abc123",
    }
    assert all(r.headers["modal-key"] == "wk-test" for r in requests)


def test_max_tokens_uses_longest_lora_context(requests):
    check("Qwen/Qwen3.5-9B", max_tokens=100_000)
    with pytest.raises(DojoConfigError, match="131072-token context"):
        check("Qwen/Qwen3.5-9B", max_tokens=131072)


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3.5-4B", "missing/model"])
def test_rejects_models_without_lora_deployment(requests, model_name):
    with pytest.raises(DojoConfigError, match="no LoRA deployment"):
        check(model_name)


def test_explains_rejected_credentials(requests):
    with pytest.raises(DojoConfigError, match="proxy-tokens allow"):
        check("Qwen/Qwen3.5-2B", api_key="wrong")


def test_rejected_server_fails_before_creating_a_run_or_app(requests, monkeypatch):
    from modal_dojo.common import config as dojo_config
    from modal_dojo.common.run import TrainingRun
    from modal_dojo.frameworks.spindle import app

    def created(*args, **kwargs):
        raise AssertionError("created a Dojo run or Modal app")

    # The default test env has no tinker-cookbook, and so no chz.
    monkeypatch.setitem(sys.modules, "chz", ModuleType("chz"))
    monkeypatch.setattr(dojo_config, "require_migrated_config", lambda: None)
    monkeypatch.setattr(TrainingRun, "save", created)
    monkeypatch.setattr(app, "build_spindle_app", created)
    config = SimpleNamespace(
        base_url=URL,
        model_name="Qwen/Qwen3.5-2B",
        max_tokens=64,
        load_checkpoint_path=None,
    )
    recipe = SimpleNamespace(rl_config=config, _api_key="wrong")
    with pytest.raises(DojoConfigError, match="rejected the request"):
        app.launch(recipe, show_output=False)


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (
            httpx.Response(404, text="Not Found"),
            "answered /api/v1/spindle/deployments with 404",
        ),
        (httpx.Response(200, text="<html>login</html>"), "unexpected response"),
    ],
)
def test_explains_servers_that_are_not_spindle(monkeypatch, response, match):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/get_server_capabilities":
            return httpx.Response(200, json={"spindle_version": "0.1.5"})
        return response

    client = functools.partial(httpx.Client, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        launcher, "httpx", SimpleNamespace(**{**vars(httpx), "Client": client})
    )
    with pytest.raises(DojoConfigError, match=match):
        check("Qwen/Qwen3.5-2B")
