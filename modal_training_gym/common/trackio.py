"""Trackio metric logging configuration and W&B compatibility adapter."""

from __future__ import annotations

import os
import secrets
import sys
import types
import uuid
from importlib import import_module
from importlib.machinery import ModuleSpec
from dataclasses import dataclass
from typing import Any, ClassVar, Self
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.metric_mirror import mirror_log
from modal_training_gym.common.metrics import MetricConfig


_DEFAULT_TRACKIO_VERSION = "0.34.0"
_DEFAULT_MODAL_APP_NAME = "training-gym-trackio"
_RUN_NAME_ENV = "TRAINING_GYM_TRACKIO_RUN_NAME"
_SHIM_MARKER = "_training_gym_trackio_adapter"


@dataclass
class TrackioConfig(MetricConfig):
    """Trackio logging configuration shared across all frameworks.

    Attributes:
        space_id: Optional Hugging Face Space id such as ``owner/trackio``.
        server_url: Optional self-hosted Trackio server URL. Leave empty to use
            hosted Trackio.
        dashboard_url: Dashboard URL. Derived from ``space_id`` or ``server_url``
            when empty.
        bucket_id: Hugging Face Bucket used by the Trackio Space.
        modal_secret_name: Modal Secret with ``HF_TOKEN`` or ``TRACKIO_WRITE_TOKEN``.
        TRACKIO_PACKAGE_VERSION: Trackio release installed in the training image
            and by ``deploy_to_modal``.
    """

    project: str = ""
    group: str = ""
    exp_name: str = ""
    disable_random_suffix: bool = True
    space_id: str = ""
    server_url: str = ""
    dashboard_url: str = ""
    bucket_id: str = ""
    modal_secret_name: str = "huggingface-secret"
    TRACKIO_PACKAGE_VERSION: str = _DEFAULT_TRACKIO_VERSION

    provider: ClassVar[str] = "trackio"  # pyright: ignore[reportIncompatibleMethodOverride]

    @classmethod
    def deploy_to_modal(
        cls,
        *,
        project: str = "",
        group: str = "",
        exp_name: str = "",
        disable_random_suffix: bool = True,
        app_name: str = _DEFAULT_MODAL_APP_NAME,
        volume_name: str = "",
        modal_secret_name: str = "",
        TRACKIO_PACKAGE_VERSION: str = _DEFAULT_TRACKIO_VERSION,
    ) -> Self:
        """Deploy a persistent Trackio server to Modal and return its config.

        The app, Volume, and write-token Secret are reused on subsequent calls
        with the same names. Ingestion and dashboard mutations require the write
        token stored in the Secret. Reads are open unless a dashboard password
        was set with ``training-gym set-password``, which gates this dashboard
        behind the same HTTP Basic Auth as the observability dashboard.
        """
        volume_name = volume_name or f"{app_name}-data"
        modal_secret_name = modal_secret_name or f"_{app_name}-write-token"
        server_url = _deploy_modal_dashboard(
            app_name=app_name,
            volume_name=volume_name,
            modal_secret_name=modal_secret_name,
            TRACKIO_PACKAGE_VERSION=TRACKIO_PACKAGE_VERSION,
        )
        return cls(
            project=project,
            group=group,
            exp_name=exp_name,
            disable_random_suffix=disable_random_suffix,
            server_url=server_url,
            dashboard_url=server_url,
            modal_secret_name=modal_secret_name,
            TRACKIO_PACKAGE_VERSION=TRACKIO_PACKAGE_VERSION,
        )

    def runtime_env(self, *, run_id: str, entity: str = "") -> dict[str, str]:
        env = super().runtime_env(run_id=run_id, entity=entity)
        for key, value in (
            (_RUN_NAME_ENV, run_id),
            ("TRACKIO_SPACE_ID", self.space_id),
            ("TRACKIO_SERVER_URL", self.server_url),
            ("TRACKIO_BUCKET_ID", self.bucket_id),
        ):
            if value:
                env[key] = value
        return env

    def url(self, *, entity: str = "", run_id: str = "") -> str | None:
        if self.dashboard_url:
            url = self.dashboard_url
        else:
            space_id = self.space_id.strip().strip("/")
            if "/" in space_id:
                url = "https://huggingface.co/spaces/" + quote(space_id, safe="/")
            elif self.server_url:
                url = self.server_url
            else:
                return None
        return _without_credentials(
            url, project=self.project or "training-gym", run_id=run_id
        )


def _without_credentials(url: str, *, project: str = "", run_id: str = "") -> str:
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().split("?", 1)[0].split("#", 1)[0]
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in {"api_key", "hf_token", "token", "write_token"}
    ]
    query_keys = {key.lower() for key, _ in query_items}
    if project and "project" not in query_keys:
        query_items.append(("project", project))
    if run_id and not query_keys.intersection({"run_ids", "runs"}):
        query_items.append(("runs", run_id))
    query = urlencode(query_items)
    return urlunsplit((parsed.scheme, host, parsed.path, query, ""))


def _deploy_modal_dashboard(
    *,
    app_name: str,
    volume_name: str,
    modal_secret_name: str,
    TRACKIO_PACKAGE_VERSION: str = _DEFAULT_TRACKIO_VERSION,
) -> str:
    import modal

    from modal_training_gym.common.config import DASHBOARD_PASSWORD_SECRET_NAME

    modal.Secret.objects.create(
        modal_secret_name,
        {"TRACKIO_WRITE_TOKEN": secrets.token_urlsafe(32)},
        allow_existing=True,
    )
    write_secret = modal.Secret.from_name(
        modal_secret_name, required_keys=["TRACKIO_WRITE_TOKEN"]
    )
    function_secrets = [write_secret]
    # Mounted only when the operator has set a password; absent it, reads are open.
    password_secret = modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME)
    try:
        password_secret.hydrate()
    except Exception:
        pass
    else:
        function_secrets.append(password_secret)
    data = modal.Volume.from_name(volume_name, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
        f"trackio=={TRACKIO_PACKAGE_VERSION}"
    )
    app = modal.App(app_name)

    @app.function(
        name="dashboard",
        image=image,
        volumes={"/data": data},
        secrets=function_secrets,
        env={"TRACKIO_DIR": "/data"},
        max_containers=1,
        scaledown_window=300,
        serialized=True,
    )
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app()
    def dashboard():
        import base64
        import binascii

        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import Response
        from trackio.server import build_starlette_app_only

        starlette_app, _ = build_starlette_app_only()
        dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")
        write_token = os.environ.get("TRACKIO_WRITE_TOKEN", "")

        def _password_ok(authorization: str | None) -> bool:
            scheme, _, encoded = (authorization or "").partition(" ")
            if scheme.lower() != "basic" or not encoded:
                return False
            try:
                decoded = base64.b64decode(encoded).decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                return False
            _user, _, supplied = decoded.partition(":")
            return secrets.compare_digest(supplied, dashboard_password)

        def _write_token_ok(supplied: str) -> bool:
            return bool(write_token) and secrets.compare_digest(supplied, write_token)

        async def require_password(request, call_next):
            if dashboard_password and not (
                _password_ok(request.headers.get("Authorization"))
                or _write_token_ok(request.headers.get("X-Trackio-Write-Token", ""))
            ):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="training-gym"'},
                )
            return await call_next(request)

        starlette_app.add_middleware(BaseHTTPMiddleware, dispatch=require_password)
        return starlette_app

    with modal.enable_output():
        app.deploy()
    server_url = dashboard.get_web_url()
    if not server_url:
        raise RuntimeError(f"Deployed {app_name!r} but no web URL was returned.")
    return server_url


def trackio_secrets(config: TrackioConfig) -> list[Any]:
    if not config.modal_secret_name or config.modal_secret_name == "huggingface-secret":
        return []
    from modal import Secret

    return [Secret.from_name(config.modal_secret_name)]


def apply_trackio_image(image: Any, config: TrackioConfig) -> Any:
    return image.uv_pip_install(f"trackio=={config.TRACKIO_PACKAGE_VERSION}")


class TrackioLookupUnknown(Exception):
    """Modal lookup of the Trackio app failed before not-found could be observed."""


def lookup_trackio_url(app_name: str = _DEFAULT_MODAL_APP_NAME) -> str | None:
    """URL of the deployed Trackio server, ``None`` if it is not deployed.

    Raises ``TrackioLookupUnknown`` when the lookup fails for any other reason.
    """
    import modal
    from modal.exception import NotFoundError

    try:
        return modal.Function.from_name(app_name, "dashboard").get_web_url()
    except NotFoundError:
        return None
    except Exception as exc:
        raise TrackioLookupUnknown from exc


def deployed_trackio_url(app_name: str = _DEFAULT_MODAL_APP_NAME) -> str | None:
    """URL of an already-deployed Trackio server on Modal, if there is one."""
    try:
        return lookup_trackio_url(app_name)
    except TrackioLookupUnknown:
        return None


def _secret_exists(name: str) -> bool:
    import modal

    try:
        modal.Secret.from_name(name).hydrate()
        return True
    except Exception:
        return False


def has_trackio_destination(config: TrackioConfig) -> bool:
    """dashboard_url is only where links point; ingestion needs a real endpoint."""
    return bool(config.space_id or config.server_url)


def resolve_trackio_destination(
    config: TrackioConfig, *, app_name: str = _DEFAULT_MODAL_APP_NAME
) -> None:
    """Point a bare ``TrackioConfig(project=...)`` at the deployed Trackio server.

    Unresolved, Trackio logs to a database local to the training container and
    the metrics die with it, so raise instead when nothing is deployed.
    """
    if has_trackio_destination(config):
        return
    url = deployed_trackio_url(app_name)
    if not url:
        raise TrainingGymConfigError(
            f"TrackioConfig(project={config.project!r}) has no destination and no "
            f"{app_name!r} server is deployed. Run TrackioConfig.deploy_to_modal() "
            "once, or set space_id= or server_url=."
        )
    secret_name = config.modal_secret_name
    if not secret_name or secret_name == "huggingface-secret":
        # The write-token name is only a convention (deploy_to_modal takes
        # modal_secret_name), so confirm it exists before adopting it.
        candidate = f"_{app_name}-write-token"
        if not _secret_exists(candidate):
            raise TrainingGymConfigError(
                f"Found the {app_name!r} Trackio server at {url} but no "
                f"{candidate!r} Secret. If it was deployed with a custom "
                "modal_secret_name, pass the config deploy_to_modal() returned, "
                "or set server_url= and modal_secret_name=."
            )
        secret_name = candidate

    # Mutate only after everything resolved, so a failed attempt stays retryable.
    config.server_url = url
    config.dashboard_url = url
    config.modal_secret_name = secret_name


def require_trackio_destination(config: TrackioConfig) -> None:
    """In-container check that a destination was resolved before launch."""
    if has_trackio_destination(config):
        return
    raise TrainingGymConfigError(
        f"TrackioConfig(project={config.project!r}) reached the training "
        "container with no destination; metrics would be lost."
    )


def preflight_trackio(config: TrackioConfig) -> str:
    try:
        import trackio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Trackio logging is enabled, but Trackio is missing from the training image."
        ) from exc
    require_trackio_destination(config)
    return ""


class _Settings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _RunProxy:
    def __init__(self, run: Any, resume_name: str) -> None:
        self._run = run
        self._resume_name = resume_name

    @property
    def id(self) -> str:
        return self._resume_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)


def install_wandb_shim() -> None:
    """Expose Trackio as the W&B subset used by Slime and Miles."""
    current = sys.modules.get("wandb")
    if current is not None and getattr(current, _SHIM_MARKER, False):
        return

    trackio: Any = import_module("trackio")

    shim: Any = types.ModuleType("wandb")
    shim.__spec__ = ModuleSpec("wandb", loader=None, is_package=True)
    shim.__path__ = []
    setattr(shim, _SHIM_MARKER, True)
    shim.run = None
    shim.config = {}
    shim.Settings = _Settings

    def init(*args: Any, **kwargs: Any) -> _RunProxy:
        project = kwargs.pop("project", args[0] if args else "") or "training-gym"
        framework_id = kwargs.pop("id", "")
        framework_name = kwargs.pop("name", "")
        requested_name = (
            os.environ.get(_RUN_NAME_ENV, "") or framework_id or framework_name
        )
        resume = kwargs.pop("resume", "allow" if requested_name else "never")
        kwargs.pop("settings", None)
        routing = {
            key: value
            for key, value in (
                ("space_id", os.environ.get("TRACKIO_SPACE_ID", "")),
                ("server_url", os.environ.get("TRACKIO_SERVER_URL", "")),
                ("bucket_id", os.environ.get("TRACKIO_BUCKET_ID", "")),
            )
            if value
        }
        config = kwargs.pop("config", None)
        if config is not None:
            config = {
                key: value
                for key, value in config.items()
                if not (isinstance(key, str) and key.startswith("_"))
            }
        run = trackio.init(
            project=project,
            name=requested_name or None,
            group=kwargs.pop("group", None),
            config=config,
            resume=resume,
            embed=False,
            **routing,
        )
        # Materialize the remote run before Slime's worker processes resume it.
        trackio.log({}, step=-1)
        proxy = _RunProxy(run, requested_name or run.name)
        shim.run = proxy
        shim.config = run.config
        return proxy

    def log(
        data: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        # trackio.log() reads the run from a ContextVar that threads started
        # after init() can't see (slime's engine-metrics thread, for one), so
        # go through the run object directly.
        run = shim.run
        if run is None:
            result = trackio.log(data, step=step)
        else:
            result = run.log(metrics=data, step=step)
        mirror_log(data, step=step, commit=commit)
        return result

    def finish(*_args: Any, **_kwargs: Any) -> Any:
        try:
            return trackio.finish()
        finally:
            shim.run = None

    def save(glob_str: str, *_args: Any, **_kwargs: Any) -> Any:
        return trackio.save(glob_str)

    def generate_id() -> str:
        return uuid.uuid4().hex[:8]

    shim.init = init
    shim.log = log
    shim.finish = finish
    shim.save = save
    shim.login = lambda **kwargs: True
    shim.define_metric = lambda *args, **kwargs: None
    shim.__getattr__ = lambda name: getattr(trackio, name)

    util: Any = types.ModuleType("wandb.util")
    util.__spec__ = ModuleSpec("wandb.util", loader=None)
    util.generate_id = generate_id
    sdk: Any = types.ModuleType("wandb.sdk")
    sdk.__spec__ = ModuleSpec("wandb.sdk", loader=None, is_package=True)
    sdk.__path__ = []
    lib: Any = types.ModuleType("wandb.sdk.lib")
    lib.__spec__ = ModuleSpec("wandb.sdk.lib", loader=None, is_package=True)
    lib.__path__ = []
    runid: Any = types.ModuleType("wandb.sdk.lib.runid")
    runid.__spec__ = ModuleSpec("wandb.sdk.lib.runid", loader=None)
    runid.generate_id = generate_id
    shim.util = util
    shim.sdk = sdk
    sdk.lib = lib
    lib.runid = runid

    sys.modules.update(
        {
            "wandb": shim,
            "wandb.util": util,
            "wandb.sdk": sdk,
            "wandb.sdk.lib": lib,
            "wandb.sdk.lib.runid": runid,
        }
    )
