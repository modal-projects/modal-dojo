"""Run a TinkerProgram against an existing or freshly deployed Spindle server."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import secrets
import sys
import uuid
from contextlib import asynccontextmanager

import httpx

from modal_dojo.common.config import modal_proxy_auth_headers
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.modal_lifecycle import stop_app_by_name

logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def shared_server(base_url, model_name, *, run, max_tokens):
    """An existing server; nothing to stop on exit."""
    api_key = os.environ["TINKER_API_KEY"]
    # The shared server is redeployed from Spindle main; record what ran.
    run.metadata.update(
        await asyncio.to_thread(
            check_server, base_url, api_key, model_name, max_tokens=max_tokens
        )
    )
    yield base_url, api_key


@asynccontextmanager
async def deployment(model_name, deploy_config, *, run, max_tokens):
    """Deploy a uniquely named LoRA Spindle server, and stop all of it on exit."""
    import modal
    from modal.config import config as modal_config
    from spindle.deployment_cli import deploy
    from spindle.deployments import DeploymentConfig
    from spindle.providers.modal.lora_pool import LoraPoolSpec

    recipe = copy.deepcopy(deploy_config)
    run_name = "dojo-spindle-" + uuid.uuid4().hex[:12]
    recipe.name = run_name
    recipe.platform["frontend"] = run_name
    environment = modal_config.get("environment")
    recipe.platform["modal"]["environment"] = environment
    # Modal-owned workspaces block unauthenticated web endpoints.
    recipe.platform["requires_proxy_auth"] = True
    # Validate existing credentials before creating apps or requesting GPUs.
    for key, required in (
        ("sampler_proxy", ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]),
        ("huggingface", ["HF_TOKEN"]),
    ):
        secret_name = recipe.platform["secrets"][key]
        if key == "huggingface" and not secret_name:
            continue
        try:
            await modal.Secret.from_name(
                secret_name, environment_name=environment, required_keys=required
            ).hydrate.aio()
        except (modal.exception.NotFoundError, modal.exception.ConflictError) as exc:
            if key == "huggingface" and isinstance(exc, modal.exception.NotFoundError):
                recipe.platform["secrets"][key] = None
                continue
            raise DojoConfigError(
                f"Spindle needs Modal secret {secret_name!r} with keys {required} "
                f"in environment {environment or 'default'!r}. Configure "
                "deploy_config.platform['secrets'] before training."
            ) from exc
    api_key = "tml-dojo-" + secrets.token_urlsafe(32)
    api_secret = run_name + "-api"
    recipe.platform["secrets"]["api"] = api_secret
    compiled = DeploymentConfig.create(recipe)
    # Only apps this run created; stop producers before the pool so they
    # cannot provision a replacement.
    names = [
        run_name,
        compiled.trainer_app_name,
        compiled.inference_app_name,
        LoraPoolSpec(compiled.definition_id).app_name,
    ]
    # Findable on the run even if the driver container is killed.
    run.metadata.update(spindle_apps=names, spindle_api_secret=api_secret)
    await run.save(is_async=True)
    await modal.Secret.objects.create.aio(
        api_secret, {"TINKER_API_KEY": api_key}, environment_name=environment
    )
    try:
        # Finish deployment before cleanup, even on cancellation.
        pending = asyncio.create_task(asyncio.to_thread(deploy, [compiled]))
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                await pending
            except Exception:
                logger.exception("Spindle deployment failed while cancelling")
            raise
        trainer_app = await modal.App.lookup.aio(
            compiled.trainer_app_name, environment_name=environment
        )
        # Spindle already tags its trainer as a training job; link the run
        # only while Modal's 8-tag cap leaves room.
        tags = await trainer_app.get_tags.aio()
        if len(tags) < 8:
            await trainer_app.set_tags.aio(
                {**tags, "training_run_id": run.training_run_id}
            )
        run.metadata["spindle_trainer_app_id"] = trainer_app.app_id
        logger.info("Spindle deployment %s in %s", run_name, environment or "default")
        server = modal.Function.from_name(
            run_name, "server", environment_name=environment
        )
        url = await server.get_web_url.aio()
        if not url:
            raise RuntimeError("Spindle deployed without an API URL")
        run.metadata.update(
            await asyncio.to_thread(
                check_server, url, api_key, model_name, max_tokens=max_tokens
            )
        )
        yield url, api_key
    finally:
        if await asyncio.to_thread(_remove_deployment, names, api_secret, environment):
            cleanup_error = RuntimeError(
                f"Spindle cleanup failed; check {', '.join(names)} and "
                f"secret {api_secret} in the logs above"
            )
            if active_error := sys.exception():
                active_error.add_note(str(cleanup_error))
            else:
                raise cleanup_error


def _remove_deployment(names, api_secret, environment) -> list[Exception]:
    """Stop the run's apps, then delete their Dicts and the API key secret."""
    import modal

    errors = []
    app_ids = []
    for name in names:
        try:
            app_ids.append(modal.App.lookup(name, environment_name=environment).app_id)
        except modal.exception.NotFoundError:
            pass
        except Exception as exc:
            errors.append(exc)
            logger.exception("Could not look up Spindle app %s", name)
    for name in names:
        try:
            stop_app_by_name(name, environment)
        except Exception as exc:
            errors.append(exc)
            logger.exception("Could not stop Spindle app %s", name)
    # Spindle keeps its state in Dicts named after the app that made them.
    prefixes = tuple(f"{app_id}-" for app_id in app_ids)
    try:
        stores = (
            modal.Dict.objects.list(environment_name=environment or "")
            if prefixes
            else []
        )
        for store in stores:
            if store.name and store.name.startswith(prefixes):
                modal.Dict.objects.delete(
                    store.name, environment_name=environment, allow_missing=True
                )
    except Exception as exc:
        errors.append(exc)
        logger.exception("Could not delete Spindle's Dicts")
    try:
        modal.Secret.objects.delete(
            api_secret, environment_name=environment, allow_missing=True
        )
    except Exception as exc:
        errors.append(exc)
        logger.exception("Could not delete Spindle secret %s", api_secret)
    return errors


async def run_program(
    program, *, run, base_url, log_dir, max_tokens=None, deploy_config=None
):
    """Run ``program`` in the Modal container ``launch()`` spawns.

    Trains against ``base_url``, or a server deployed from ``deploy_config``.
    """
    import tinker

    from modal_dojo.tinker import TinkerContext

    from .reporting import run_lifecycle

    # The driver owns its single-use container, so every ServiceClient in it,
    # tinker-cookbook's included, reaches Spindle through the Modal proxy.
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
    server = (
        shared_server(base_url, program.model_name, run=run, max_tokens=max_tokens)
        if deploy_config is None
        else deployment(
            program.model_name, deploy_config, run=run, max_tokens=max_tokens
        )
    )
    async with run_lifecycle(run), server as (base_url, api_key):
        os.environ["TINKER_BASE_URL"] = base_url
        os.environ["TINKER_API_KEY"] = api_key
        with tinker.ServiceClient() as client:
            await program.run(client, TinkerContext(run, log_dir))
    return run
