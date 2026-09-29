"""Recipe-driven custom model deployments."""

from __future__ import annotations

import asyncio
import inspect
import threading

from modal.experimental import list_deployed_apps
from pydantic import BaseModel, ConfigDict, model_validator

from modal_dojo.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_megatron_checkpoint_to_hf,
)
from modal_dojo.common.errors import TrainingGymConfigError
from modal_dojo.common.openai_messages import _messages_to_openai
from modal_dojo.common.ids import create_hash
from modal_dojo.common.modal_urls import modal_app_dashboard_url
from modal_dojo.common.models import ModelConfig
from modal_dojo.deploy_recipes.base import DeployRecipeType
from modal_dojo.deploy_recipes.sglang_recipe import SglangRecipe
from modal_dojo.deploy_recipes.vllm_recipe import VllmRecipe


def _run_coro(coro):
    """Run a coroutine to completion, even from inside a running event loop.

    ``asyncio.run`` raises when an event loop is already running (e.g. in a
    Jupyter notebook), so in that case we run the coroutine on a dedicated
    worker thread with its own loop. If ``coro`` is already a resolved value
    (some Modal versions return the URL synchronously), it is returned as-is.
    """
    if not inspect.isawaitable(coro):
        return coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            result["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _modal_proxy_auth_headers() -> dict[str, str]:
    """Headers to authenticate against Modal endpoints behind proxy auth.

    Reads the Modal proxy-auth token pair (``wk-``/``ws-``) from ``MODAL_KEY`` /
    ``MODAL_SECRET`` and returns them as ``Modal-Key`` / ``Modal-Secret`` headers.
    Falls back to the pair persisted in ``~/.training-gym.toml`` (written by
    ``training-gym setup``) when the env vars are unset. Returns an empty dict
    when neither source provides them, so endpoints without proxy auth are
    unaffected.
    """
    from modal_dojo.common.config import modal_proxy_auth_headers

    return modal_proxy_auth_headers()


def _raise_for_proxy_auth(status_code: int, url: str) -> None:
    """Turn a 401 into an actionable proxy-auth hint.

    SGLang endpoints are public by default (``unauthenticated=True``). When
    an endpoint was served with ``unauthenticated=False``, a 401 almost
    always means the ``MODAL_KEY`` / ``MODAL_SECRET`` proxy-auth token pair
    is missing from the environment (so :func:`_modal_proxy_auth_headers`
    returned no headers) rather than a real authorization problem — surface
    that instead of a bare ``HTTPError``/``TimeoutError``. No-op for any
    other status.
    """
    if status_code != 401:
        return
    sent_auth = bool(_modal_proxy_auth_headers())
    detail = (
        "the MODAL_KEY / MODAL_SECRET proxy-auth tokens are not set in this environment"
        if not sent_auth
        else "the MODAL_KEY / MODAL_SECRET tokens were sent but rejected (expired "
        "or wrong workspace)"
    )
    raise RuntimeError(
        f"401 Unauthorized from {url} — this endpoint is behind Modal proxy auth "
        f"and {detail}. Create a proxy-auth token pair at "
        "https://modal.com/settings/proxy-auth-tokens and export MODAL_KEY (wk-…) "
        "and MODAL_SECRET (ws-…) in the shell that runs the eval/serve. For calls "
        "issued from remote workers (e.g. a custom rm/reward function), also "
        "forward the pair into the worker via a modal.Secret. SGLang endpoints "
        "are public by default; pass CustomDeployment.launch(..., "
        "unauthenticated=False) to "
        "require proxy auth."
    )


class _CrashloopDetector:
    """Crashloop heuristic over periodic container-count samples.

    Zero containers is tolerated for ``cold_start_grace`` before any container
    has been seen running (image pull, weight download, and server launch all
    happen in that window), but only for ``restart_grace`` once one has come
    up — a container that appeared and then vanished is a real restart cycle.
    """

    def __init__(self, *, cold_start_grace: int, restart_grace: int) -> None:
        self._cold_start_grace = cold_start_grace
        self._restart_grace = restart_grace
        self._zero_since: float | None = None
        self._seen_running = False

    def observe(self, containers: int, now: float) -> tuple[int, str] | None:
        """Record a container-count sample; returns ``(elapsed, phase)`` on crashloop.

        Negative counts are transient failures querying Modal app state and are
        ignored rather than counted as zero containers.
        """
        if containers < 0:
            return None
        if containers > 0:
            self._seen_running = True
            self._zero_since = None
            return None
        if self._zero_since is None:
            self._zero_since = now
        elapsed = int(now - self._zero_since)
        grace = self._restart_grace if self._seen_running else self._cold_start_grace
        if elapsed < grace:
            return None
        phase = (
            "came up and then died"
            if self._seen_running
            else f"never reached a running state within {grace}s"
        )
        return elapsed, phase


class CustomDeployment(BaseModel):
    """A model deployed with an SGLang or vLLM recipe.

    Attributes:
        deployment_id: Stable deployment ID.
        model: Served model configuration.
        recipe: Serving recipe.
        app_name: Modal app name.
        served_model_name: Model name exposed to clients.
        unauthenticated: Whether the endpoint accepts unauthenticated requests.
        modal_app_id: Modal app ID.
        modal_app_url: Dashboard URL.
        url: Endpoint URL.
    """

    deployment_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: ModelConfig
    recipe: VllmRecipe | SglangRecipe | None = None
    app_name: str
    served_model_name: str
    unauthenticated: bool = True

    modal_app_id: str = ""
    modal_app_url: str = ""
    url: str

    @model_validator(mode="before")
    @classmethod
    def _parse_deployment(cls, value: object) -> object:
        """Flatten records written by the legacy deployment API."""
        if not isinstance(value, dict) or "deployment_config" not in value:
            return value

        payload = dict(value)
        legacy = payload.pop("deployment_config")
        if not isinstance(legacy, dict):
            return payload
        for field in ("model", "recipe", "app_name", "served_model_name"):
            if field in legacy:
                payload.setdefault(field, legacy[field])
        payload.setdefault("unauthenticated", legacy.get("unauthenticated", True))
        if isinstance(payload.get("model"), dict):
            payload["model"] = ModelConfig(**payload["model"])
        return payload

    @classmethod
    def launch(
        cls,
        model: ModelConfig | str,
        checkpoint: Checkpoint | None = None,
        *,
        recipe: VllmRecipe | SglangRecipe | None = None,
        app_name: str | None = None,
        served_model_name: str | None = None,
        unauthenticated: bool = True,
    ) -> "CustomDeployment":
        """Deploy ``model`` with an SGLang or vLLM serving recipe.

        Args:
            model:
                Model configuration or Hugging Face model name.
            checkpoint:
                Training checkpoint to convert and serve.
            recipe:
                SGLang or vLLM serving recipe.
            app_name:
                Modal app name derived from ``model`` when omitted.
            served_model_name:
                Model name exposed to clients.
            unauthenticated:
                Whether the endpoint accepts requests without proxy credentials.

        Returns:
            The deployed ``CustomDeployment`` handle.
        """
        model = ModelConfig(model_name=model) if isinstance(model, str) else model

        if recipe is None:
            recipe = SglangRecipe()

        if (
            checkpoint is not None
            and checkpoint.checkpoint_type == CheckpointType.megatron
        ):
            checkpoint = convert_megatron_checkpoint_to_hf(
                checkpoint=checkpoint,
                model=model,
                recipe=recipe,
            )

        if checkpoint is not None and checkpoint.path:
            model_path = checkpoint.path
        else:
            model_path = model.model_path or model.model_name
        if not model_path:
            raise TrainingGymConfigError(
                f"{type(model).__name__} has no model path to serve. "
                "Set model_path or model_name."
            )

        default_name_src = model.model_name or model_path
        default_slug = (
            default_name_src.rstrip("/").split("/")[-1].replace("_", "-").lower()
        )
        app_name = app_name or f"{default_slug}-serve"
        served_model_name = served_model_name or default_slug
        deployment_id = create_hash(
            model.model_name,
            checkpoint.path if checkpoint is not None else "",
            f"{type(recipe).__name__}:{recipe._recipe_type.value}",
            app_name,
            model_path,
        )
        checkpoints_volume = (
            checkpoint.checkpoints_volume_name
            if checkpoint is not None and checkpoint.checkpoints_volume_name
            else getattr(model, "checkpoints_volume_name", None)
        )
        checkpoints_mount_path = (
            checkpoint.checkpoints_mount_path
            if checkpoint is not None and checkpoint.checkpoints_mount_path
            else getattr(model, "checkpoints_mount_path", None)
        )

        if isinstance(recipe, SglangRecipe):
            from modal_dojo.deploy_recipes.sglang_recipe.serve_sglang import (
                build_sglang_serve_app,
            )

            app = build_sglang_serve_app(
                recipe=recipe,
                app_name=app_name,
                model_path=model_path,
                served_model_name=served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
                unauthenticated=unauthenticated,
            )
        elif isinstance(recipe, VllmRecipe):
            from modal_dojo.deploy_recipes.vllm_recipe.serve_vllm import (
                build_vllm_serve_app,
            )

            app = build_vllm_serve_app(
                recipe=recipe,
                app_name=app_name,
                model_path=model_path,
                served_model_name=served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
                unauthenticated=unauthenticated,
            )
        else:
            raise TrainingGymConfigError(
                f"Unsupported deploy recipe: {type(recipe).__name__}"
            )

        app.deploy(
            environment_name=recipe.environment_name,
            strategy=recipe.deploy_strategy,
        )

        server_attr = (
            "SGLangEndpoint"
            if recipe._recipe_type == DeployRecipeType.SGLANG
            else "Server"
        )
        server = getattr(app, server_attr, None)
        if server is None:
            raise RuntimeError(
                f"Deployed {app_name!r} but could not resolve "
                f"{server_attr} server handle."
            )
        url = _run_coro(server.get_url())
        modal_app_id = app.app_id
        if not url:
            raise RuntimeError(f"Deployed {app_name!r} but no web URL was returned.")
        if not modal_app_id:
            raise RuntimeError(
                f"Deployed {app_name!r} but no Modal app id was returned."
            )

        deployment = cls(
            deployment_id=deployment_id,
            model=model,
            recipe=recipe,
            app_name=app_name,
            served_model_name=served_model_name,
            unauthenticated=unauthenticated,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_dashboard_url(modal_app_id),
            url=url,
        )
        return deployment

    # TODO(atoniolo76): A future PR should update all existing tutorials to
    # use this new function while getting rid of the old generate.
    def chat(
        self,
        messages: list[dict],
        timeout: int = 120,
        max_attempts: int = 4,
        **extra,
    ) -> dict:
        """Send one OpenAI-compatible chat-completion request.

        Args:
            messages:
                OpenAI-compatible chat messages.
            timeout:
                Timeout in seconds for each request.
            max_attempts:
                Maximum request attempts for transient failures.
            extra:
                Additional Chat Completions request fields.

        Returns:
            The assistant ``message`` dict.
        """
        import time

        import requests

        body = {
            "model": self.served_model_name,
            "messages": _messages_to_openai(messages),
            **extra,
        }
        transient_status_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=body,
                    timeout=timeout,
                    headers=_modal_proxy_auth_headers(),
                )
                if (
                    resp.status_code in transient_status_codes
                    and attempt < max_attempts
                ):
                    print(
                        f"Transient generation error {resp.status_code} from {self.url}; "
                        f"retrying ({attempt}/{max_attempts})..."
                    )
                    time.sleep(min(2 * attempt, 5))
                    continue
                _raise_for_proxy_auth(resp.status_code, self.url)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= max_attempts:
                    raise
                print(
                    f"Transient generation transport error from {self.url}: {exc}; "
                    f"retrying ({attempt}/{max_attempts})..."
                )
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(
            f"Failed to generate from {self.url} after {max_attempts} attempts"
        )

    def generate(
        self,
        prompt: str | list[dict],
        **kwargs,
    ) -> str:
        """Send a chat-completion request.

        Returns:
            The assistant text, or ``reasoning_content`` when ``content`` is ``None``.
        """
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        message = self.chat(
            messages,
            **kwargs,
        )
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return message.get("reasoning_content", "")
        return str(content)

    def _start_log_tailer(self) -> "threading.Thread | None":
        """Spawn a daemon thread that streams deployed-app logs to stdout.

        Returns the thread (or None if we can't resolve the app).
        """
        import modal

        app_name = self.app_name
        env_name = self.recipe.environment_name if self.recipe is not None else None

        def _tail() -> None:
            try:
                app = modal.App.lookup(app_name, environment_name=env_name)
            except Exception:
                return

            async def _stream() -> None:
                async for entry in app.logs.stream.aio():
                    print(entry.message, end="", flush=True)

            try:
                asyncio.run(_stream())
            except Exception:
                pass

        t = threading.Thread(target=_tail, daemon=True)
        t.start()
        return t

    def wait_until_ready(self, timeout: int = 15 * 60) -> None:
        """Wait until the deployment can serve traffic.

        Args:
            timeout:
                Maximum number of seconds to wait.
        """
        import time

        import requests

        app_name = self.app_name
        logs_hint = (
            f"`modal app logs {self.modal_app_id}`"
            if self.modal_app_id
            else f"`modal app logs {app_name}`"
        )
        print(f"Waiting for {app_name!r} — {self.modal_app_url}")

        log_thread = self._start_log_tailer()

        # A freshly-deployed SGLang container legitimately reports 0 running
        # containers for the whole cold-start window: pulling the (multi-GB)
        # image, downloading weights, and launching the server all happen before
        # it serves its first request. Modal allows the container `startup_timeout`
        # to reach readiness, so the crashloop heuristic must not fire inside that
        # window — otherwise a slow-but-healthy boot is misreported as
        # "crashlooping". Only flag a crashloop quickly once we've actually seen a
        # container come up and then disappear (a real restart cycle).
        recipe = self.recipe
        startup_timeout = recipe.startup_timeout if recipe is not None else 20 * 60
        # Cap the cold-start grace at the overall deadline — otherwise it always
        # exceeds `timeout` and the loop hits TimeoutError before the crashloop
        # check can fire, so cold-start crashloop detection never engages.
        detector = _CrashloopDetector(
            cold_start_grace=min(startup_timeout, timeout),
            restart_grace=60,
        )

        deadline = time.time() + timeout
        modal_poll_interval = 20
        last_modal_poll = 0.0
        last_probe = "no response yet"

        try:
            while time.time() < deadline:
                resp = None
                probe = last_probe
                try:
                    resp = requests.get(
                        f"{self.url}/v1/models",
                        timeout=10,
                        headers=_modal_proxy_auth_headers(),
                    )
                    if resp.ok and resp.json().get("data"):
                        return
                    probe = f"HTTP {resp.status_code}"
                except requests.ConnectionError as exc:
                    probe = f"connection error ({type(exc).__name__})"
                except Exception as exc:
                    probe = f"{type(exc).__name__}: {exc}"

                if probe != last_probe:
                    last_probe = probe
                    print(f"[deployment] probe {self.url}/v1/models: {probe}")

                # A 401 is an auth problem, not a cold-start (loading returns
                # 502/503) — fail fast with a proxy-auth hint instead of polling
                # to the timeout. Checked outside the try so it isn't swallowed.
                if resp is not None:
                    _raise_for_proxy_auth(resp.status_code, self.url)

                now = time.time()
                if now - last_modal_poll >= modal_poll_interval:
                    last_modal_poll = now
                    containers = self._modal_container_count()
                    if containers is None:
                        raise RuntimeError(
                            f"Modal app {app_name!r} is not in a deployed state — "
                            f"the deploy likely failed or was stopped. Check {self.modal_app_url} "
                            f"and {logs_hint}."
                        )
                    crashloop = detector.observe(containers, now)
                    if crashloop is not None:
                        elapsed, phase = crashloop
                        raise RuntimeError(
                            f"Modal app {app_name!r} has had 0 running containers for "
                            f"~{elapsed}s ({phase}, last probe: {last_probe}). Containers "
                            f"are most likely crashlooping on startup (OOM, missing weights, "
                            f"bad config, etc.). Inspect logs with {logs_hint} or open "
                            f"{self.modal_app_url}."
                        )

                time.sleep(5)
        finally:
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=2)

        raise TimeoutError(
            f"{self.url} not ready after {timeout}s (last probe: {last_probe}). "
            f"Inspect logs with {logs_hint} or open {self.modal_app_url}."
        )

    def _modal_container_count(self) -> int | None:
        """Container count for this deployment, or None if app isn't DEPLOYED."""
        try:
            apps = _run_coro(list_deployed_apps())
        except Exception as exc:
            print(f"[deployment] couldn't query Modal app state: {exc!r}")
            return -1
        for app in apps:
            if app.app_id == self.modal_app_id or app.name == self.app_name:
                return app.containers
        return None
