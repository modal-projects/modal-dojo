"""Run tinker-cookbook RL against a Spindle server."""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import httpx

from modal_dojo.common.config import modal_proxy_auth_headers
from modal_dojo.common.errors import DojoConfigError

logger = logging.getLogger(__name__)
# tinker-cookbook hooks are process-wide.
_active_run = threading.Lock()


async def check_server(base_url: str, api_key: str, config) -> dict[str, str | None]:
    """Fail unless the server has a LoRA deployment that fits ``config``.

    Returns the server's identity for the run record: the shared server is
    redeployed from Spindle main, so one config can run on several versions.
    """
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={**modal_proxy_auth_headers(), "X-API-Key": api_key},
        timeout=30,
    ) as client:
        try:
            capabilities, deployments = await asyncio.gather(
                client.get("/api/v1/get_server_capabilities"),
                client.get("/api/v1/spindle/deployments"),
            )
        except httpx.TransportError as exc:
            raise DojoConfigError(f"Could not reach Spindle at {base_url}: {exc}")
    for response in (capabilities, deployments):
        if response.status_code in (401, 403):
            raise DojoConfigError(
                f"Spindle at {base_url} rejected the request "
                f"({response.status_code}). Check TINKER_API_KEY, and that the "
                "Modal proxy token in MODAL_KEY/MODAL_SECRET is allowed for the "
                "server's environment (`modal workspace proxy-tokens allow "
                "<wk-token-id> <environment>`)."
            )
        response.raise_for_status()
    served = deployments.json()["deployments"]
    lora = [
        deployment
        for deployment in served
        if deployment["base_model"] == config.model_name
        and deployment["parameterization"] == "lora"
    ]
    if not lora:
        available = sorted(
            {d["base_model"] for d in served if d["parameterization"] == "lora"}
        )
        raise DojoConfigError(
            f"Spindle at {base_url} has no LoRA deployment of "
            f"{config.model_name}; it serves {available}"
        )
    # Spindle routes a base model to its longest-context deployment.
    context = max(deployment["max_context_length"] for deployment in lora)
    if config.max_tokens >= context:
        raise DojoConfigError(
            f"max_tokens must leave room for prompts in {config.model_name}'s "
            f"{context}-token context"
        )
    info = capabilities.json()
    return {
        "spindle_base_url": base_url,
        "spindle_version": info.get("spindle_version"),
        "spindle_commit": info.get("spindle_commit"),
    }


async def train(config, *, api_key):
    from tinker_cookbook.rl import train as rl_train
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from .reporting import DojoReporter

    if not _active_run.acquire(blocking=False):
        raise DojoConfigError(
            "Only one SpindleRecipe may train per Python process at a time"
        )
    original_tinker = rl_train.tinker
    try:
        # Fail before creating a Dojo run.
        tokenizer = get_tokenizer(config.model_name)
        server = await check_server(config.base_url, api_key, config)
        async with DojoReporter(config, server) as reporter:
            await _run_tinker_cookbook(config, api_key, tokenizer, reporter)
        return reporter.run
    finally:
        rl_train.tinker = original_tinker
        _active_run.release()


async def _run_tinker_cookbook(config, api_key, tokenizer, reporter):
    import tinker
    from tinker_cookbook.rl import train as rl_train
    from .trajectories import capture_trajectories

    with reporter.tinker_cookbook_hooks(rl_train):
        clients = []

        def service_client(**kwargs):
            client = tinker.ServiceClient(
                **{
                    **kwargs,
                    "base_url": config.base_url,
                    "api_key": api_key,
                    # tinker ignores the older default_headers name.
                    "default_headers_overrides": modal_proxy_auth_headers(),
                }
            )
            clients.append(client)
            return client

        # Scope client routing to tinker-cookbook.
        rl_train.tinker = SimpleNamespace(
            **{
                **vars(tinker),
                "ServiceClient": service_client,
            }
        )
        status = "errored"
        try:
            async with capture_trajectories(
                config.log_path, tokenizer, on_group=reporter.capture_group
            ):
                await rl_train.main(config)
            status = "success"
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        finally:
            for client in clients:
                try:
                    await asyncio.wait_for(client.close(status), timeout=30)
                except Exception:
                    logger.exception("Could not close tinker-cookbook's Spindle client")
