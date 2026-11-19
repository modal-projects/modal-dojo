"""Select a bundled Spindle config and provision a reusable model deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import time

import httpx

from modal_dojo.common.errors import DojoConfigError

logger = logging.getLogger(__name__)
SPINDLE_VERSION = "0.1.8"
SPINDLE_REQUIREMENT = (
    "modal-spindle @ git+https://github.com/modal-projects/spindle.git@"
    "d23774980fb804eeca378930d19adcaede67c2f0"
)
SERVER_NAME = "dojo-spindle-v" + SPINDLE_VERSION.replace(".", "-")
DEPLOYMENTS_PATH = "/api/v1/spindle/deployments"
_FRONTEND_FIELDS = {
    "platform",
    "session_idle_timeout_s",
    "pool_idle_timeout_s",
    "sweep_interval_s",
}


@dataclass(frozen=True)
class Requirements:
    model_name: str
    max_tokens: int
    lora_rank: int
    context_length: int | None = None
    overrides: dict = field(default_factory=dict)

    def fits(self, row: dict) -> bool:
        context = row["max_context_length"]
        return (
            row["base_model"] == self.model_name
            and row["parameterization"] == "lora"
            and context > self.max_tokens
            and context >= (self.context_length or 0)
        )


def select_config(requirements: Requirements):
    """Choose the smallest supported context, then apply hardware overrides."""
    try:
        from spindle import __version__
        from spindle.deploy_wizard import bundled_presets
        from spindle.configuration import BaseConfig
        from spindle.deployments import DeploymentConfig
    except ImportError as exc:
        raise DojoConfigError(
            "Provisioning Spindle needs Python 3.12 and modal-dojo[spindle-deploy]. "
            "Install with `uv sync --extra spindle-deploy` in this repository, "
            "or connect to an existing deployment without GPU overrides."
        ) from exc

    if __version__ != SPINDLE_VERSION:
        raise DojoConfigError(
            f"Provisioning requires Spindle {SPINDLE_VERSION}, but {__version__} is installed. "
            "Install the pinned spindle-deploy extra before deploying."
        )

    candidates = [
        preset.config
        for preset in bundled_presets()
        if requirements.fits(
            {
                "base_model": preset.model,
                "parameterization": preset.mode,
                "max_context_length": preset.context,
            }
        )
    ]
    if not candidates:
        raise DojoConfigError(
            f"No bundled Spindle LoRA config supports {requirements.model_name!r} "
            f"with context_length={requirements.context_length!r} and "
            f"max_tokens={requirements.max_tokens}. Deploy a supported custom "
            "config with Spindle, then provide its server URL and key."
        )
    preset = min(candidates, key=lambda c: (c.max_context_length, c.name))
    config = BaseConfig(**vars(preset), overrides=requirements.overrides)
    try:
        compiled = DeploymentConfig.create(config)
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise DojoConfigError(f"Invalid Spindle GPU configuration: {exc}") from exc
    max_rank = compiled.inference_settings.get("max_lora_rank")
    if max_rank is not None and requirements.lora_rank > max_rank:
        raise DojoConfigError(
            f"LoRA rank {requirements.lora_rank} exceeds {config.name}'s "
            f"maximum of {max_rank}."
        )
    return config


def deployment_name(config) -> str:
    """Identify the config consistently across explicit URLs and app discovery."""
    settings = {k: v for k, v in vars(config).items() if k not in {"name", "platform"}}
    digest = hashlib.sha256(
        json.dumps([SPINDLE_VERSION, settings], sort_keys=True).encode()
    ).hexdigest()[:12]
    # Spindle scopes GPU app names with its frontend name.
    return f"dojo-{digest}"


def request_json(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.TransportError as exc:
        raise DojoConfigError(f"Could not reach the Tinker server: {exc}") from exc
    if response.status_code in (401, 403):
        raise DojoConfigError(
            "The server rejected the credentials. Check its TINKER_API_KEY and, "
            "for protected Modal endpoints, the proxy token's environment access."
        )
    if not response.is_success:
        raise DojoConfigError(
            f"Server request {method} {path} failed (HTTP {response.status_code}). "
            "For backend provisioning, the Spindle server must support the "
            "deployment API. No replacement server was created."
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise DojoConfigError("The server returned an invalid JSON response.") from exc
    if not isinstance(body, dict):
        raise DojoConfigError(f"Expected a JSON object from {path}.")
    return body


def ensure_backend(
    client: httpx.Client, requirements: Requirements, *, provision: bool = True
) -> dict | None:
    rows = request_json(client, "GET", DEPLOYMENTS_PATH)["deployments"]
    matches = [row for row in rows if requirements.fits(row)]
    config = None
    if requirements.overrides or not matches:
        config = select_config(requirements)
        name = deployment_name(config)
        # Reuse configs created with the earlier server-specific suffix as well.
        matches = [
            row
            for row in matches
            if row["name"] == name or row["name"].startswith(name + "-")
        ]
    if matches:
        selected = min(
            matches,
            key=lambda row: (
                row.get("apps", "ready") != "ready",
                row["max_context_length"],
                row["name"],
            ),
        )
        if not provision or selected.get("apps", "ready") == "ready":
            return selected
        name = selected["name"]
        if selected.get("apps") in {"missing", "failed"}:
            request_json(
                client, "POST", DEPLOYMENTS_PATH, json={"name": name, "base": name}
            )
    else:
        if not provision:
            return None
        logger.info("Adding Spindle backend %s", name)
        request_json(
            client,
            "POST",
            DEPLOYMENTS_PATH,
            json={
                "name": name,
                "recipe": {
                    key: value
                    for key, value in vars(config).items()
                    if key not in _FRONTEND_FIELDS and key != "name"
                },
            },
        )

    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        rows = request_json(client, "GET", DEPLOYMENTS_PATH)["deployments"]
        selected = next((row for row in rows if row["name"] == name), None)
        if selected is None:
            raise DojoConfigError(
                f"Spindle backend {name!r} disappeared during deployment."
            )
        status = selected.get("apps")
        if status == "ready":
            return selected
        if status in (None, "failed", "missing"):
            raise DojoConfigError(
                f"Spindle backend {name!r} could not be provisioned (apps={status!r}). "
                "Check the server's deployment logs; the existing server was retained."
            )
        time.sleep(5)
    raise DojoConfigError(f"Timed out waiting for Spindle backend {name!r} to deploy.")
