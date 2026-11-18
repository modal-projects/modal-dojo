"""Contracts for ``modal_dojo.remote``: the code Modal containers import.

These functions are registered without ``serialized=True``, so a container
finds them by importing their module. Two things have to hold for that to
work, and neither is exercised by an ordinary unit test:

- the module imports in an image that has none of Modal Dojo's dependencies;
- the callable is a module-level object Modal can look up by name.
"""

from __future__ import annotations

import json
import pathlib
import pkgutil
import subprocess
import sys

import modal
import pytest

import modal_dojo.remote
from modal_dojo.deploy_recipes.sglang_recipe import SglangRecipe
from modal_dojo.deploy_recipes.sglang_recipe.serve_sglang import build_sglang_serve_app
from modal_dojo.deploy_recipes.vllm_recipe import VllmRecipe
from modal_dojo.deploy_recipes.vllm_recipe.serve_vllm import build_vllm_serve_app
from modal_dojo.remote import serving

REMOTE_MODULES = sorted(
    f"modal_dojo.remote.{info.name}"
    for info in pkgutil.iter_modules(modal_dojo.remote.__path__)
)

# What a bare Modal image provides. Anything else a remote module imports at
# module scope would fail in the Trackio, vLLM, and CPU-converter images.
ALLOWED_THIRD_PARTY = {"modal", "modal_dojo"}


def test_remote_package_has_modules() -> None:
    assert REMOTE_MODULES


@pytest.mark.parametrize("module_name", REMOTE_MODULES)
def test_remote_module_imports_without_dojo_dependencies(module_name: str) -> None:
    probe = f"""
import importlib, json, sys
before = set(sys.modules)
import modal  # present in every Modal container
baseline = set(sys.modules)
importlib.import_module({module_name!r})
added = sorted(
    name for name in set(sys.modules) - baseline
    if name.split(".")[0] not in sys.stdlib_module_names
)
print(json.dumps(added))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        check=True,
    )
    added = json.loads(result.stdout.strip().splitlines()[-1])
    heavy = [
        name
        for name in added
        if name.split(".")[0] not in ALLOWED_THIRD_PARTY
        or (name.startswith("modal_dojo.") and not name.startswith("modal_dojo.remote"))
    ]
    assert not heavy, (
        f"{module_name} imports {heavy} at module scope; move those imports "
        "inside the function that needs them"
    )


@pytest.mark.parametrize("module_name", REMOTE_MODULES)
def test_remote_callables_are_module_level(module_name: str) -> None:
    module = sys.modules.get(module_name) or __import__(module_name, fromlist=["_"])
    for name, obj in vars(module).items():
        if getattr(obj, "__module__", None) != module_name or name.startswith("_"):
            continue
        assert obj.__qualname__ == name, f"{module_name}.{name} is not module-level"


class _RecordingApp:
    """Stands in for ``modal.App`` and records each ``app.server`` call."""

    servers: list[tuple[type, dict]] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def server(self, **kwargs):
        def register(cls: type) -> type:
            _RecordingApp.servers.append((cls, kwargs))
            return cls

        return register


@pytest.fixture
def recorded_servers(monkeypatch) -> list[tuple[type, dict]]:
    monkeypatch.setattr(_RecordingApp, "servers", [])
    monkeypatch.setattr(modal, "App", _RecordingApp)
    return _RecordingApp.servers


def _serve_spec(kwargs: dict) -> dict:
    """The serve spec a server registered with ``kwargs`` reads at startup."""
    assert "serialized" not in kwargs
    assert kwargs["include_source"] is False
    return json.loads(kwargs["env"][serving.SERVE_SPEC_ENV])


def test_vllm_serve_app_registers_the_importable_server_class(
    recorded_servers,
) -> None:
    # The same module-level class backs every app, so each build's parameters
    # must be carried by the function's environment, not by the class.
    for model_path in ("org/first", "org/second"):
        build_vllm_serve_app(
            recipe=VllmRecipe(),
            app_name="test-vllm-serve",
            model_path=model_path,
            served_model_name="model",
            revision="abc123",
        )

    (first_cls, first), (second_cls, second) = recorded_servers
    assert first_cls is second_cls is serving.Server
    assert _serve_spec(first)["model_path"] == "org/first"
    assert _serve_spec(second) == {
        "model_path": "org/second",
        "served_model_name": "model",
        "port": 8000,
        "n_gpu": 1,
        "extra_args": ["--revision", "abc123"],
    }


def test_sglang_serve_app_registers_the_importable_server_class(
    recorded_servers,
) -> None:
    build_sglang_serve_app(
        recipe=SglangRecipe(),
        app_name="test-sglang-serve",
        model_path="org/model",
        served_model_name="model",
        revision="abc123",
    )

    ((cls, kwargs),) = recorded_servers
    assert cls is serving.SGLangEndpoint
    spec = _serve_spec(kwargs)
    assert spec["model_path"] == "org/model"
    assert spec["server_args"]["--served-model-name"] == "model"
    assert spec["server_args"]["--revision"] == "abc123"
