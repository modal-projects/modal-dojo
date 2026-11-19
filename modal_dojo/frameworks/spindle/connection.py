"""Environment-scoped Spindle discovery and persistent connection credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import sys
from urllib.parse import urlsplit

import click

from modal_dojo.common.errors import DojoConfigError

from .deployment import SERVER_NAME, Requirements, deployment_name, select_config

DEFAULT_SECRET = "dojo-tinker-default"


def validate_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DojoConfigError("base_url must be an HTTP(S) server URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DojoConfigError(
            "base_url must not contain credentials, a query, or a fragment."
        )
    return url.rstrip("/")


def proxy_headers_for(url: str) -> dict[str, str]:
    from modal_dojo.common.config import modal_proxy_auth_headers

    host = urlsplit(url).hostname or ""
    return modal_proxy_auth_headers() if host.endswith(".modal.run") else {}


@dataclass(frozen=True)
class Connection:
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    secret_name: str | None = None
    environment: str | None = None
    app_name: str | None = None

    def driver_secrets(self):
        import modal

        if self.secret_name:
            return [
                modal.Secret.from_name(
                    self.secret_name,
                    environment_name=self.environment,
                    required_keys=["TINKER_API_KEY"],
                )
            ]
        return [modal.Secret.from_dict({"TINKER_API_KEY": self.api_key})]


def _secret_exists(name: str, environment: str, required_keys: list[str]) -> bool:
    import modal

    try:
        modal.Secret.from_name(
            name, environment_name=environment, required_keys=required_keys
        ).hydrate()
    except modal.exception.NotFoundError:
        return False
    return True


def lookup_server(environment: str) -> Connection | None:
    import modal

    try:
        server = modal.Function.from_name(
            SERVER_NAME, "server", environment_name=environment
        )
        url = server.get_web_url()
    except modal.exception.NotFoundError:
        return None
    if not url:
        raise DojoConfigError(f"Spindle app {SERVER_NAME!r} has no server URL.")
    platform = modal.Function.from_name(
        SERVER_NAME, "deployed_platform", environment_name=environment
    ).remote()
    secret_name = platform["secrets"]["api"]
    if not _secret_exists(secret_name, environment, ["TINKER_API_KEY"]):
        raise DojoConfigError(f"Spindle's API-key Secret {secret_name!r} is missing.")
    click.echo(
        f"Connecting to {SERVER_NAME} in environment {environment!r}: {url}", err=True
    )
    return Connection(
        url, secret_name=secret_name, environment=environment, app_name=SERVER_NAME
    )


def resolve_connection(
    base_url: str | None,
    api_key: str | None,
    requirements: Requirements,
    *,
    yes: bool = False,
    environment: str | None = None,
) -> Connection:
    from modal.config import config

    environment = environment or config.get("environment") or "main"
    if base_url:
        if not api_key:
            raise DojoConfigError(
                "Provide api_key or TINKER_API_KEY for the specified base_url."
            )
        return Connection(validate_url(base_url), api_key, environment=environment)
    if _secret_exists(
        DEFAULT_SECRET, environment, ["TINKER_BASE_URL", "TINKER_API_KEY"]
    ):
        click.echo(
            f"Using saved Tinker connection {DEFAULT_SECRET} in {environment!r}.",
            err=True,
        )
        return Connection(secret_name=DEFAULT_SECRET, environment=environment)
    existing = lookup_server(environment)
    if existing is not None:
        return existing
    deployment = select_config(requirements)
    _confirm_deploy(deployment, environment, yes=yes)
    # A second caller may have finished setup while this caller confirmed.
    existing = lookup_server(environment)
    if existing is not None:
        return existing
    return deploy_server(deployment, environment)


def _confirm_deploy(config, environment: str, *, yes: bool):
    description = (
        f"Deploy {SERVER_NAME} in Modal environment {environment!r} for {config.model} "
        f"({config.max_context_length} context, "
        f"{config.trainer_nodes * config.trainer_gpus_per_node} x {config.trainer_gpu} trainer, "
        f"{config.inference_gpus_per_node} x {config.inference_gpu} per sampler)? "
        "The server will remain available for future runs."
    )
    if yes:
        click.echo(description, err=True)
        return
    if not sys.stdin.isatty():
        raise DojoConfigError(
            "No saved connection or compatible Spindle server exists in this environment. "
            "Provide base_url and api_key, approve deployment with .train(yes=True) / "
            ".launch(yes=True)."
        )
    if not click.confirm(description, default=False, err=True):
        raise DojoConfigError("Spindle deployment cancelled. No server was created.")


def deploy_server(config, environment: str) -> Connection:
    """Deploy once under stable names; retain apps, storage, and credentials."""
    import modal
    from spindle.deployment_cli import deploy
    from spindle.deployments import DeploymentConfig

    from modal_dojo.common.config import modal_proxy_auth_headers

    if sys.version_info[:2] != (3, 12):
        raise DojoConfigError("Deploying Spindle requires Python 3.12.")
    config.name = deployment_name(config)
    config.platform["frontend"] = SERVER_NAME
    config.platform["modal"]["environment"] = environment
    config.platform["requires_proxy_auth"] = True
    proxy = modal_proxy_auth_headers()
    if not proxy:
        raise DojoConfigError(
            "Deploying a protected Spindle server requires MODAL_KEY and MODAL_SECRET "
            "for this environment. Configure them with `modal-dojo setup`."
        )
    proxy_secret = config.platform["secrets"]["sampler_proxy"]
    modal.Secret.objects.create(
        proxy_secret,
        {
            "MODAL_PROXY_TOKEN_ID": proxy["Modal-Key"],
            "MODAL_PROXY_TOKEN_SECRET": proxy["Modal-Secret"],
        },
        environment_name=environment,
        allow_existing=True,
    )
    if not _secret_exists(
        proxy_secret, environment, ["MODAL_PROXY_TOKEN_ID", "MODAL_PROXY_TOKEN_SECRET"]
    ):
        raise DojoConfigError(
            f"Spindle's sampler proxy Secret {proxy_secret!r} is missing "
            f"from environment {environment!r}."
        )
    hf_secret = config.platform["secrets"].get("huggingface")
    if hf_secret and not _secret_exists(hf_secret, environment, ["HF_TOKEN"]):
        config.platform["secrets"]["huggingface"] = None
    api_secret = SERVER_NAME + "-api"
    config.platform["secrets"]["api"] = api_secret
    # create-if-missing retains the existing key on retries and concurrent setup.
    modal.Secret.objects.create(
        api_secret,
        {"TINKER_API_KEY": "tml-dojo-" + secrets.token_urlsafe(32)},
        environment_name=environment,
        allow_existing=True,
    )
    with modal.enable_output():
        deploy([DeploymentConfig.create(config)])
    connection = lookup_server(environment)
    if connection is None:
        raise DojoConfigError(
            "Spindle deployment finished without a discoverable server."
        )
    return connection
