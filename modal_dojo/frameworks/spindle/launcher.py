"""Resolve a model backend and run its Tinker client in the Dojo driver."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os

import httpx

from modal_dojo.common.errors import DojoConfigError

from .connection import proxy_headers_for, validate_url
from .deployment import (
    SERVER_NAME,
    SPINDLE_VERSION,
    Requirements,
    ensure_backend,
    request_json,
)


def check_server(
    base_url, api_key, model_name, *, max_tokens=None, requirements=None, scope=None
):
    """Validate any Tinker service; provision and select only on Spindle."""
    base_url = validate_url(base_url)
    requirements = requirements or Requirements(model_name, max_tokens or 0, 32)
    with httpx.Client(
        base_url=base_url,
        headers={**proxy_headers_for(base_url), "X-API-Key": api_key},
        timeout=60,
    ) as client:
        info = request_json(client, "GET", "/api/v1/get_server_capabilities")
        if scope == SERVER_NAME and info.get("spindle_version") != SPINDLE_VERSION:
            raise DojoConfigError(
                f"Managed app {SERVER_NAME!r} is not serving Spindle {SPINDLE_VERSION}. "
                "Check that deployment, or provide an explicit server URL and key."
            )
        if info.get("spindle_version"):
            selected = ensure_backend(client, requirements, scope=scope or base_url)
            return {
                "spindle_base_url": base_url,
                "spindle_version": info["spindle_version"],
                "spindle_commit": info.get("spindle_commit"),
                "spindle_deployment": selected["name"],
                "spindle_context_length": selected["max_context_length"],
            }
        if requirements.overrides:
            raise DojoConfigError(
                "GPU overrides require a Spindle server that supports provisioning."
            )
        supported = [
            row
            for row in info.get("supported_models", [])
            if row.get("model_name") == model_name
        ]
        if not supported:
            raise DojoConfigError(
                f"The Tinker server does not advertise model {model_name!r}."
            )
        context = supported[0].get("max_context_length")
        if context is not None and (
            context < (requirements.context_length or 0)
            or context <= requirements.max_tokens
        ):
            raise DojoConfigError(
                f"The server's {context}-token context does not fit this run."
            )
        return {"spindle_base_url": base_url, "tinker_provider": "external"}


@contextmanager
def routed_clients(base_url: str, model_name: str, deployment: str | None):
    """Route SDK model creation without changing Cookbook's tokenizer model."""
    import tinker

    original = tinker.ServiceClient

    class ServiceClient(original):
        def __init__(self, *args, default_headers_overrides=None, **kwargs):
            target = kwargs.get("base_url") or base_url
            kwargs.setdefault("base_url", base_url)
            self._dojo_route = (
                deployment if validate_url(str(target)) == base_url else None
            )
            super().__init__(
                *args,
                default_headers_overrides={
                    **proxy_headers_for(str(target)),
                    **(default_headers_overrides or {}),
                },
                **kwargs,
            )

        def _route(self, model):
            return (
                self._dojo_route if self._dojo_route and model == model_name else model
            )

        def create_lora_training_client(self, base_model, *args, **kwargs):
            return super().create_lora_training_client(
                self._route(base_model), *args, **kwargs
            )

        async def create_lora_training_client_async(self, base_model, *args, **kwargs):
            return await super().create_lora_training_client_async(
                self._route(base_model), *args, **kwargs
            )

        def create_sampling_client(
            self, model_path=None, base_model=None, *args, **kwargs
        ):
            return super().create_sampling_client(
                model_path, self._route(base_model), *args, **kwargs
            )

        async def create_sampling_client_async(
            self, model_path=None, base_model=None, *args, **kwargs
        ):
            return await super().create_sampling_client_async(
                model_path, self._route(base_model), *args, **kwargs
            )

    tinker.ServiceClient = ServiceClient
    try:
        yield ServiceClient
    finally:
        tinker.ServiceClient = original


async def run_program(
    program, *, run, base_url, log_dir, max_tokens=None, requirements=None, scope=None
):
    from .context import TinkerContext
    from .reporting import run_lifecycle

    async with run_lifecycle(run):
        base_url = base_url or os.environ.get("TINKER_BASE_URL")
        api_key = os.environ.get("TINKER_API_KEY")
        if not base_url or not api_key:
            raise DojoConfigError(
                "The connection Secret must supply TINKER_BASE_URL and TINKER_API_KEY."
            )
        base_url = validate_url(base_url)
        os.environ["TINKER_BASE_URL"] = base_url
        server = await asyncio.to_thread(
            check_server,
            base_url,
            api_key,
            program.model_name,
            max_tokens=max_tokens,
            requirements=requirements,
            scope=scope,
        )
        run.metadata.update(server)
        await run.save(is_async=True)
        with routed_clients(
            base_url, program.model_name, server.get("spindle_deployment")
        ) as client_type:
            with client_type(api_key=api_key) as client:
                await program.run(client, TinkerContext(run, log_dir))
    return run
