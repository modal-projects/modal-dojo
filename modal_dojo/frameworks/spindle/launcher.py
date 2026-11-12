"""Run a TinkerProgram against a Spindle server, inside the driver container."""

from __future__ import annotations

import asyncio
import os

import httpx

from modal_dojo.common.config import modal_proxy_auth_headers
from modal_dojo.common.errors import DojoConfigError


def check_server(
    base_url: str, api_key: str, model_name: str, *, max_tokens: int | None = None
) -> dict[str, str | None]:
    """Fail unless the server has a LoRA deployment of ``model_name`` that fits.

    Returns the server's identity for the run record: the shared server is
    redeployed from Spindle main, so one config can run on several versions.
    """
    with httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={**modal_proxy_auth_headers(), "X-API-Key": api_key},
        timeout=30,
    ) as client:
        try:
            capabilities = client.get("/api/v1/get_server_capabilities")
            deployments = client.get("/api/v1/spindle/deployments")
        except httpx.TransportError as exc:
            raise DojoConfigError(
                f"Could not reach Spindle at {base_url}: {exc}"
            ) from exc
    for response in (capabilities, deployments):
        if response.status_code in (401, 403):
            raise DojoConfigError(
                f"Spindle at {base_url} rejected the request "
                f"({response.status_code}). Check TINKER_API_KEY, and that the "
                "Modal proxy token in MODAL_KEY/MODAL_SECRET is allowed for the "
                "server's environment (`modal workspace proxy-tokens allow "
                "<wk-token-id> <environment>`)."
            )
        if not response.is_success:
            raise DojoConfigError(
                f"Spindle at {base_url} answered {response.request.url.path} with "
                f"{response.status_code}; is it a Spindle server?"
            )
    try:
        served = deployments.json()["deployments"]
        info = capabilities.json()
    except (ValueError, KeyError) as exc:
        raise DojoConfigError(
            f"Spindle at {base_url} sent an unexpected response: {exc!r}"
        ) from exc
    lora = [
        deployment
        for deployment in served
        if deployment["base_model"] == model_name
        and deployment["parameterization"] == "lora"
    ]
    if not lora:
        available = sorted(
            {d["base_model"] for d in served if d["parameterization"] == "lora"}
        )
        raise DojoConfigError(
            f"Spindle at {base_url} has no LoRA deployment of "
            f"{model_name}; it serves {available}"
        )
    # Spindle routes a base model to its longest-context deployment.
    context = max(deployment["max_context_length"] for deployment in lora)
    if max_tokens is not None and max_tokens >= context:
        raise DojoConfigError(
            f"max_tokens must leave room for prompts in {model_name}'s "
            f"{context}-token context"
        )
    return {
        "spindle_base_url": base_url,
        "spindle_version": info.get("spindle_version"),
        "spindle_commit": info.get("spindle_commit"),
    }


async def run_program(program, *, run, base_url, log_dir, max_tokens=None):
    """Run ``program`` in the Modal container ``launch()`` spawns."""
    import tinker

    from modal_dojo.tinker import TinkerContext

    from .reporting import run_lifecycle

    # The driver owns its single-use container, so every ServiceClient in it,
    # tinker-cookbook's included, reaches Spindle through the Modal proxy.
    os.environ["TINKER_BASE_URL"] = base_url
    proxy_headers = modal_proxy_auth_headers()

    # A subclass, not a partial, so user code can still subclass and isinstance it.
    class ServiceClient(tinker.ServiceClient):
        def __init__(self, *args, default_headers_overrides=None, **kwargs):
            super().__init__(
                *args,
                default_headers_overrides={
                    **proxy_headers,
                    **(default_headers_overrides or {}),
                },
                **kwargs,
            )

    tinker.ServiceClient = ServiceClient
    async with run_lifecycle(run):
        # The shared server is redeployed from Spindle main; record what ran.
        run.metadata.update(
            await asyncio.to_thread(
                check_server,
                base_url,
                os.environ["TINKER_API_KEY"],
                program.model_name,
                max_tokens=max_tokens,
            )
        )
        with tinker.ServiceClient() as client:
            await program.run(client, TinkerContext(run, log_dir))
    return run
