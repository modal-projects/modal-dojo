from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from typing import Any

import httpx
import modal

from modal_dojo.common.checkpoint import (
    Checkpoint,
    convert_megatron_checkpoint_to_hf,
)
from modal_dojo.common.config import modal_proxy_auth_headers
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.openai_messages import _messages_to_openai
from modal_dojo.model import ModelConfig


def _stop_endpoint(endpoint_name: str, environment: str | None) -> None:
    stop = [
        sys.executable,
        "-m",
        "modal",
        "endpoint",
        "stop",
        endpoint_name,
        "--yes",
    ]
    if environment:
        stop.extend(["--env", environment])
    stopped = subprocess.run(
        stop, check=False, capture_output=True, text=True, timeout=120
    )
    if stopped.returncode != 0:
        text = f"{stopped.stdout or ''}{stopped.stderr or ''}"
        if not re.search(
            r"endpoint '[^']+' not found|endpoint .+ is already stopped",
            text,
            flags=re.IGNORECASE,
        ):
            sys.stdout.write(stopped.stdout or "")
            sys.stderr.write(stopped.stderr or "")
            raise subprocess.CalledProcessError(
                stopped.returncode,
                stop,
                output=stopped.stdout,
                stderr=stopped.stderr,
            )


class Endpoint:
    """Controls a [Modal Endpoint](https://modal.com/docs/guide/endpoints) that
    persists until stopped.

    Attributes:
        url: Base URL of the endpoint.
        endpoint_name: Modal Endpoint name.
        model_name: Base model ID sent in request bodies.
        requires_proxy_auth: Whether a proxy token is required to use the endpoint.
    """

    url: str
    endpoint_name: str
    model_name: str
    requires_proxy_auth: bool

    def __init__(
        self,
        url: str,
        *,
        endpoint_name: str,
        model_name: str,
        requires_proxy_auth: bool,
    ):
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.requires_proxy_auth = requires_proxy_auth

    @classmethod
    def launch(
        cls,
        model: ModelConfig | str,
        checkpoint: Checkpoint | None = None,
        *,
        endpoint_name: str | None = None,
        unauthenticated: bool = True,
        routing_region: str | None = None,
        environment: str | None = None,
        colocate_compute: bool = False,
        wait_timeout_sec: float = 15 * 60,
        recreate_if_existing: bool = False,
    ) -> "Endpoint":
        """Deploy ``model`` without waiting for readiness.

        Args:
            model:
                Model configuration or Hugging Face model name.
            checkpoint:
                Training checkpoint to convert and serve.
            endpoint_name:
                Endpoint name. Derived from the configuration when omitted.
            unauthenticated:
                Whether the endpoint accepts requests without proxy credentials.
            routing_region:
                Endpoint traffic region.
            environment:
                Modal environment for the endpoint.
            colocate_compute:
                Whether to place compute in the routing region.
            wait_timeout_sec:
                Maximum time to wait for an endpoint URL.
            recreate_if_existing:
                Whether to replace an endpoint with the same name.

        Returns:
            The deployed ``Endpoint`` handle.

        Raises:
            TimeoutError:
                The endpoint does not publish a URL before ``wait_timeout_sec``.
        """
        if checkpoint:
            model_config = (
                model
                if isinstance(model, ModelConfig)
                else ModelConfig(model_name=model)
            )
            checkpoint = convert_megatron_checkpoint_to_hf(checkpoint, model_config)

        model_name = model if isinstance(model, str) else model.model_name

        if not endpoint_name:
            spec = {
                "model": model_name,
                "routing_region": routing_region,
                "unauthenticated": unauthenticated,
            }

            if checkpoint:
                spec.update(
                    {
                        "checkpoint_run": checkpoint.training_run_id,
                        "checkpoint_name": checkpoint.name,
                    }
                )
            if colocate_compute:
                spec["colocate_compute"] = True

            digest = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            endpoint_name = f"training-gym-{digest}"

        endpoint = cls(
            "",
            endpoint_name=endpoint_name,
            model_name=model_name,
            requires_proxy_auth=not unauthenticated,
        )
        if recreate_if_existing:
            _stop_endpoint(endpoint_name, environment)

        command = [
            sys.executable,
            "-m",
            "modal",
            "endpoint",
            "create",
            "--name",
            endpoint_name,
            "--model",
            model_name,
        ]

        if unauthenticated:
            command.append("--unauthenticated")
        if routing_region:
            command.extend(["--routing-region", routing_region])
        if environment:
            command.extend(["--env", environment])
        if colocate_compute:
            command.append("--colocate-compute")

        if checkpoint:
            command.extend(["--custom-volume-name", checkpoint.checkpoints_volume_name])
            command.extend(["--custom-volume-path", checkpoint.path_relative_to_volume])

        created = subprocess.run(
            command, check=False, timeout=120, capture_output=True, text=True
        )
        if created.returncode != 0:
            text = f"{created.stdout or ''}{created.stderr or ''}"
            if "already exists" not in text.lower():
                sys.stdout.write(created.stdout or "")
                sys.stderr.write(created.stderr or "")
                raise subprocess.CalledProcessError(
                    created.returncode,
                    command,
                    output=created.stdout,
                    stderr=created.stderr,
                )

        server = modal.Server.from_name(
            f"ep-{endpoint_name}", "Server", environment_name=environment
        )
        deadline = time.monotonic() + wait_timeout_sec
        while time.monotonic() < deadline:
            try:
                raw = server.get_url()
            except modal.exception.NotFoundError:
                raw = None
            if raw:
                endpoint.url = raw.rstrip("/")
                return endpoint
            time.sleep(1)
        else:
            raise TimeoutError(
                f"Timed out waiting for a URL for endpoint {endpoint_name!r}"
            )

    def stop(self) -> None:
        """Stop this endpoint and terminate its containers."""
        _stop_endpoint(self.endpoint_name, None)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.requires_proxy_auth:
            headers = modal_proxy_auth_headers()
            if not headers:
                raise DojoConfigError(
                    "Proxy authentication requires an HTTPS Modal URL and "
                    "MODAL_KEY and MODAL_SECRET."
                )
        return headers

    def wait_until_ready(self, timeout: float = 15 * 60) -> None:
        """Wait until the endpoint can serve traffic.

        Args:
            timeout:
                Maximum number of seconds to wait.

        Raises:
            DojoConfigError:
                Required proxy credentials are unavailable.
            RuntimeError:
                The endpoint rejects proxy credentials.
            TimeoutError:
                The endpoint is not ready before ``timeout``.
        """
        last_error: Exception | None = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.url}/v1/models",
                    headers=self._headers(),
                    timeout=30,
                    follow_redirects=False,
                )
                if response.status_code == 200:
                    return
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"Endpoint {self.endpoint_name} rejected proxy authentication. "
                        "Run `modal-dojo set-proxy-auth` and retry."
                    )
                if response.status_code not in {404, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                last_error = RuntimeError(
                    f"Endpoint {self.endpoint_name} readiness returned HTTP {response.status_code}"
                )
            except httpx.RequestError as exc:
                last_error = exc
            time.sleep(2)
        raise TimeoutError(
            f"Timed out waiting for endpoint {self.endpoint_name} at {self.url} to become ready"
        ) from last_error

    def chat(
        self,
        messages: list[dict[str, Any]],
        timeout: int = 120,
        max_attempts: int = 4,
        **extra: Any,
    ) -> dict:
        """Send a chat-completion request.

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

        Raises:
            DojoConfigError:
                Required proxy credentials are unavailable.
            RuntimeError:
                The endpoint rejects proxy credentials.
            httpx.HTTPError:
                The request fails.
        """
        url = f"{self.url}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": _messages_to_openai(messages),
            **extra,
        }

        headers = self._headers()
        transient = {429, 500, 502, 503, 504}
        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(
                    url,
                    json=body,
                    timeout=timeout,
                    headers=headers,
                    follow_redirects=False,
                )
                if resp.status_code in transient and attempt < max_attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                if resp.status_code in {401, 403}:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from {url}. Proxy credentials were "
                        "rejected. Refresh them with `modal workspace proxy-tokens "
                        "create` and export MODAL_KEY and MODAL_SECRET."
                    )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except httpx.RequestError:
                if attempt >= max_attempts:
                    raise
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(
            f"Chat completions exhausted {max_attempts} attempts at {url}"
        )
