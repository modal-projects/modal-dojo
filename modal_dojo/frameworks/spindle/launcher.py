"""Provision a uniquely named LoRA deployment using Spindle's own deploy path."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import secrets
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from modal_dojo.common.errors import DojoConfigError

logger = logging.getLogger(__name__)
# Cookbook exposes module-level hooks. Reject overlap rather than route another
# run's traffic or trajectories through the active run's credentials.
_active_run = threading.Lock()


def _stop_app(name: str, environment: str | None) -> None:
    import modal

    try:
        app = modal.App.lookup(name, environment_name=environment)
    except modal.exception.NotFoundError:
        return
    command = [sys.executable, "-m", "modal", "app", "stop", "--yes", app.app_id]
    if environment:
        command.extend(["--env", environment])
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stderr or result.stdout
    if result.returncode and "App is already stopped" not in output:
        raise RuntimeError(f"Could not stop {name}: {output}")


@asynccontextmanager
async def deployment(config, *, metric_tags=None):
    import modal
    from spindle.deployment_cli import deploy
    from spindle.deployments import DeploymentConfig
    from spindle.providers.modal.lora_pool import LoraPoolSpec

    recipe = copy.deepcopy(config)
    run_name = "dojo-spindle-" + uuid.uuid4().hex[:12]
    recipe.name = run_name
    recipe.platform["frontend"] = run_name
    environment = recipe.platform["modal"]["environment"] or os.getenv(
        "MODAL_ENVIRONMENT"
    )
    recipe.platform["modal"]["environment"] = environment
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
    # Names are unique to this invocation, including the lazily created LoRA pool.
    # Stop producers before the pool so they cannot provision a replacement.
    names = [
        run_name,
        compiled.trainer_app_name,
        compiled.inference_app_name,
        LoraPoolSpec(compiled.definition_id).app_name,
    ]
    await modal.Secret.objects.create.aio(
        api_secret, {"TINKER_API_KEY": api_key}, environment_name=environment
    )
    try:
        # Shield deploy and wait for it on cancellation before attempting cleanup:
        # a worker thread could otherwise create an app after we stopped it.
        pending = asyncio.create_task(asyncio.to_thread(deploy, [compiled]))
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            try:
                await pending
            except Exception:
                logger.exception("Spindle deployment failed while cancelling")
            raise
        from modal_dojo.common import COMMON_DOJO_TAGS

        trainer_app = await modal.App.lookup.aio(
            compiled.trainer_app_name, environment_name=environment
        )
        await trainer_app.set_tags.aio(
            {
                **await trainer_app.get_tags.aio(),
                **COMMON_DOJO_TAGS,
                "_modal_framework": "spindle",
                **(metric_tags or {}),
            }
        )
        logger.info("Spindle deployment %s in %s", run_name, environment or "default")
        server = modal.Function.from_name(
            run_name, "server", environment_name=environment
        )
        url = await server.get_web_url.aio()
        if not url:
            raise RuntimeError("Spindle deployed without an API URL")
        yield url, api_key
    finally:
        errors = []
        for name in names:
            try:
                await asyncio.to_thread(_stop_app, name, environment)
            except Exception as exc:
                errors.append(exc)
                logger.exception("Could not stop Spindle app %s", name)
        try:
            await modal.Secret.objects.delete.aio(
                api_secret, environment_name=environment, allow_missing=True
            )
        except Exception as exc:
            errors.append(exc)
        if errors:
            cleanup_error = ExceptionGroup(
                "Spindle cleanup failed; see app names in logs", errors
            )
            if active_error := sys.exception():
                active_error.add_note(str(cleanup_error))
            else:
                raise cleanup_error


async def train(config, deploy_config) -> None:
    import chz
    import tinker
    from tinker_cookbook.rl import train as rl_train
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from .trajectories import capture_trajectories

    if not _active_run.acquire(blocking=False):
        raise DojoConfigError(
            "Only one SpindleRecipe may train per Python process at a time"
        )
    original_tinker = rl_train.tinker
    try:
        # Resolve the tokenizer before deploying GPUs; preserve the caller's config.
        tokenizer = get_tokenizer(config.model_name)
        from modal_dojo.common import modal_tag_value

        metric_tags = {}
        if config.wandb_project:
            metric_tags = {
                "_modal_metric_provider": "wandb",
                "_modal_metric_project": modal_tag_value(config.wandb_project),
            }
        async with deployment(deploy_config, metric_tags=metric_tags) as (url, api_key):
            clients = []

            def service_client(**kwargs):
                client = tinker.ServiceClient(
                    **{**kwargs, "base_url": url, "api_key": api_key}
                )
                clients.append(client)
                return client

            # Patch only cookbook's module reference, not the SDK or process env.
            rl_train.tinker = SimpleNamespace(
                **{
                    **vars(tinker),
                    "ServiceClient": service_client,
                }
            )
            status = "errored"
            try:
                async with capture_trajectories(config.log_path, tokenizer):
                    await rl_train.main(chz.replace(config, base_url=url))
                status = "success"
            except asyncio.CancelledError:
                status = "interrupted"
                raise
            finally:
                for client in clients:
                    try:
                        await asyncio.wait_for(client.close(status), timeout=30)
                    except tinker.NotFoundError:
                        # This Spindle revision lacks the SDK's session-finish
                        # route. close() still releases local holders in finally;
                        # the owned deployment is stopped immediately below.
                        pass
                    except Exception:
                        logger.exception("Could not close cookbook's Spindle client")
    finally:
        rl_train.tinker = original_tinker
        _active_run.release()
