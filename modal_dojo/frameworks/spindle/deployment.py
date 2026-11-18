"""Select a bundled Spindle config and provision a reusable model deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import time

from modal_dojo.common.errors import DojoConfigError

logger = logging.getLogger(__name__)
SPINDLE_VERSION = "0.1.8"
SPINDLE_REQUIREMENT = (
    "modal-spindle @ git+https://github.com/modal-projects/spindle.git@"
    "16ce201dae9862347cb17437fa0cfed3e3db313e"
)
SERVER_NAME = "dojo-spindle-v" + SPINDLE_VERSION.replace(".", "-")
DEPLOYMENTS_PATH = "/api/v1/spindle/deployments"


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
        from spindle.deploy_wizard import bundled_presets
        from spindle.configuration import BaseConfig
        from spindle.deployments import DeploymentConfig
    except ImportError as exc:
        raise DojoConfigError(
            "Provisioning Spindle needs Python 3.12 and modal-dojo[spindle-deploy]. "
            "Install with `uv sync --extra spindle-deploy` in this repository, "
            "or connect to an existing deployment without GPU overrides."
        ) from exc

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


def deployment_name(config, scope: str) -> str:
    """Stable across retries; scoped so two frontends never share GPU app names."""
    settings = {k: v for k, v in vars(config).items() if k not in {"name", "platform"}}
    digest = hashlib.sha256(
        json.dumps([SPINDLE_VERSION, settings], sort_keys=True).encode()
    ).hexdigest()[:12]
    server = hashlib.sha256(scope.encode()).hexdigest()[:6]
    # Spindle prefixes this with '<frontend>-inference-'; Modal caps app names.
    return f"dojo-{digest}-{server}"


def request_json(client, method: str, path: str, **kwargs):
    import httpx

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
        return response.json()
    except ValueError as exc:
        raise DojoConfigError("The server returned an invalid JSON response.") from exc


def ensure_backend(client, requirements: Requirements, *, scope: str) -> dict:
    rows = request_json(client, "GET", DEPLOYMENTS_PATH)["deployments"]
    matches = [row for row in rows if requirements.fits(row)]
    config = None
    if requirements.overrides or not matches:
        config = select_config(requirements)
        name = deployment_name(config, scope)
        # The same server can be reached by app discovery or an explicit URL.
        # Compare the config fingerprint; the suffix only isolates app names.
        config_name = name.rsplit("-", 1)[0]
        matches = [
            row for row in matches if row["name"].rsplit("-", 1)[0] == config_name
        ]
    if matches:
        selected = min(
            matches, key=lambda row: (row["max_context_length"], row["name"])
        )
        if selected.get("apps", "ready") == "ready":
            return selected
        name = selected["name"]
        if selected.get("apps") in {"missing", "failed"}:
            request_json(
                client, "POST", DEPLOYMENTS_PATH, json={"name": name, "base": name}
            )
    else:
        logger.info("Adding Spindle backend %s", name)
        request_json(
            client,
            "POST",
            DEPLOYMENTS_PATH,
            json={"name": name, "recipe": vars(config)},
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
