"""SpindleRecipe's preflight against a running Spindle server."""

from __future__ import annotations

import asyncio
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

    client = functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handle))
    monkeypatch.setattr(
        launcher, "httpx", SimpleNamespace(**{**vars(httpx), "AsyncClient": client})
    )
    monkeypatch.setattr(
        launcher, "modal_proxy_auth_headers", lambda: {"Modal-Key": "wk-test"}
    )
    return seen


def check(model_name, max_tokens=64, api_key="tml-test"):
    config = SimpleNamespace(model_name=model_name, max_tokens=max_tokens)
    return asyncio.run(launcher.check_server(URL, api_key, config))


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


def test_rejected_server_fails_before_creating_a_dojo_run(requests, monkeypatch):
    from modal_dojo.frameworks.spindle.reporting import DojoReporter

    async def create_run(self):
        raise AssertionError("created a Dojo run")

    # Stand in for tinker-cookbook, which the default test env doesn't install.
    rl = ModuleType("tinker_cookbook.rl")
    rl.train = SimpleNamespace(tinker=None)
    tokenizer_utils = ModuleType("tinker_cookbook.tokenizer_utils")
    tokenizer_utils.get_tokenizer = lambda model_name: None
    for module in (ModuleType("tinker_cookbook"), rl, tokenizer_utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(DojoReporter, "__aenter__", create_run)
    config = SimpleNamespace(
        base_url=URL,
        model_name="Qwen/Qwen3.5-2B",
        learning_rate=1e-5,
        lora_rank=32,
        loss_fn="importance_sampling",
        max_tokens=64,
        max_steps=10,
        recipe_name="test",
        renderer_name="qwen3_5_disable_thinking",
        dataset_builder=object(),
        log_path="/tmp/dojo-spindle-test",
    )
    with pytest.raises(DojoConfigError, match="rejected the request"):
        asyncio.run(launcher.train(config, api_key="wrong"))
