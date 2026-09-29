"""User-local config persisted at ``~/.training-gym.toml``.

Populated by ``modal-dojo setup``; read by the slime launcher (and any other
caller) to look up where to POST phase reports and other client-side defaults.
"""

from __future__ import annotations

import os
import tomllib
from json import JSONDecodeError, loads
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modal_dojo.common.dashboard import (
    DashboardLookupUnknown,
    deployed_dashboard_url,
)


CONFIG_PATH = Path.home() / ".training-gym.toml"
MODAL_CONFIG_PATH = Path(
    os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")
)

_dashboard_requires_proxy_auth = False
_dashboard_trajectory_viewer: str | None = None
DASHBOARD_PROXY_AUTH_PATH = "/api/proxy-auth"
DASHBOARD_VERSION_PATH = "/api/version"

# Holds DASHBOARD_PASSWORD. An empty value means the dashboard is open (no
# auth) — that's the default so existing deployments keep working untouched.
# Set a real value via ``modal-dojo set-password``.
DASHBOARD_PASSWORD_SECRET_NAME = "_training-gym-dashboard-password"


def set_dashboard_requires_proxy_auth(value: bool) -> None:
    """Set the mode used when `_dashboard` next registers its web function."""
    global _dashboard_requires_proxy_auth
    _dashboard_requires_proxy_auth = value


def dashboard_requires_proxy_auth() -> bool:
    """Return the proxy-auth mode for the next dashboard module import."""
    return _dashboard_requires_proxy_auth


def set_dashboard_trajectory_viewer(path: str | None) -> None:
    """Set the custom Svelte trajectory viewer for the next dashboard build."""
    global _dashboard_trajectory_viewer
    _dashboard_trajectory_viewer = path


def get_dashboard_trajectory_viewer() -> str | None:
    """Return the configured custom trajectory viewer, if one is configured."""
    if _dashboard_trajectory_viewer:
        return _dashboard_trajectory_viewer
    dashboard = load_config().get("dashboard")
    if isinstance(dashboard, dict):
        path = dashboard.get("trajectory_viewer")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return None


def load_config() -> dict[str, Any]:
    """Return the parsed ``~/.training-gym.toml``, or ``{}`` if missing."""
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def save_dashboard_url(url: str, *, proxy_auth: bool | None = None) -> None:
    """Persist the deployed dashboard URL and optional proxy-auth mode."""
    config = load_config()
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}
    dashboard["url"] = url
    if proxy_auth is not None:
        dashboard["proxy_auth"] = proxy_auth
    config["dashboard"] = dashboard
    CONFIG_PATH.write_text(_render(config))


def save_dashboard_trajectory_viewer(path: str | None) -> None:
    """Persist or clear the custom trajectory viewer path."""
    config = load_config()
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}
    if path:
        dashboard["trajectory_viewer"] = path
    else:
        dashboard.pop("trajectory_viewer", None)
    if dashboard:
        config["dashboard"] = dashboard
    else:
        config.pop("dashboard", None)
    CONFIG_PATH.write_text(_render(config))


class DashboardVersionUnknown(Exception):
    """The live dashboard exists but ``/api/version`` could not be read."""


def get_dashboard_version(url: str) -> str | None:
    """Return the live dashboard version, or ``None`` if it answered without one.

    Raises ``DashboardVersionUnknown`` when the request fails before that can
    be observed (timeout, 401, network, 5xx).
    """
    request = Request(
        url.rstrip("/") + DASHBOARD_VERSION_PATH,
        headers=modal_proxy_auth_headers(),
    )
    try:
        with urlopen(request, timeout=5) as response:
            value = loads(response.read())
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise DashboardVersionUnknown from exc
    except (URLError, OSError) as exc:
        raise DashboardVersionUnknown from exc
    except (JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, str) else None


def get_dashboard_url() -> str | None:
    """Return the saved dashboard base URL, or ``None``."""
    dashboard = load_config().get("dashboard")
    if isinstance(dashboard, dict):
        url = dashboard.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def get_dashboard_proxy_auth() -> bool | None:
    """Return the deployed dashboard's proxy-auth mode, if known.

    The live endpoint is authoritative. Modal itself returns 403 before a
    proxy-authenticated dashboard request reaches FastAPI, so that status also
    identifies an authenticated deployment. The persisted mode remains a
    fallback for older or temporarily unreachable dashboards.
    """
    dashboard = load_config().get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}

    persisted = dashboard.get("proxy_auth")
    if not isinstance(persisted, bool):
        persisted = None

    try:
        url = dashboard.get("url") or deployed_dashboard_url()
    except DashboardLookupUnknown:
        url = dashboard.get("url")
    if isinstance(url, str) and url.strip():
        request = Request(
            url.strip().rstrip("/") + DASHBOARD_PROXY_AUTH_PATH,
            headers=modal_proxy_auth_headers(),
        )
        try:
            with urlopen(request, timeout=5) as response:
                value = loads(response.read())
                if isinstance(value, bool):
                    if value:
                        print("The deployed dashboard uses proxy authentication.")
                    else:
                        print(
                            "The deployed dashboard does not use proxy authentication."
                        )
                    return value
        except HTTPError as exc:
            if exc.code in {401, 403}:
                print("The deployed dashboard appears to use proxy authentication.")
                return True
            elif exc.code != 404:
                raise
        except (JSONDecodeError, OSError, URLError, UnicodeDecodeError):
            pass

    if persisted is not None:
        print("Unable to reach existing dashboard.")
        if persisted:
            print("The last deploy from this computer used proxy authentication.")
        else:
            print(
                "The last deploy from this computer did not use proxy authentication."
            )

    return persisted


PROXY_AUTH_SECTION = "proxy_auth"


def get_proxy_auth() -> tuple[str, str]:
    """Return the ``(MODAL_KEY, MODAL_SECRET)`` pair saved under ``[proxy_auth]``.

    Returns empty strings for any value that is missing or blank.
    """
    section = load_config().get(PROXY_AUTH_SECTION)
    if isinstance(section, dict):
        key = str(section.get("key") or "").strip()
        secret = str(section.get("secret") or "").strip()
        return key, secret
    return "", ""


def save_proxy_auth(key: str, secret: str) -> None:
    """Persist the proxy-auth token pair under ``[proxy_auth]``."""
    config = load_config()
    config[PROXY_AUTH_SECTION] = {"key": key.strip(), "secret": secret.strip()}
    CONFIG_PATH.write_text(_render(config))


def load_proxy_auth() -> bool:
    """Populate ``MODAL_KEY`` / ``MODAL_SECRET`` from ``~/.training-gym.toml``.

    Dotenv-style: real environment variables always win and are never
    overwritten; only unset ones are filled in from the saved config. Returns
    ``True`` when both end up set in the environment.
    """
    have_key = bool(os.environ.get("MODAL_KEY", "").strip())
    have_secret = bool(os.environ.get("MODAL_SECRET", "").strip())
    if have_key and have_secret:
        return True

    key, secret = get_proxy_auth()
    if key and not have_key:
        os.environ["MODAL_KEY"] = key
    if secret and not have_secret:
        os.environ["MODAL_SECRET"] = secret
    return bool(
        os.environ.get("MODAL_KEY", "").strip()
        and os.environ.get("MODAL_SECRET", "").strip()
    )


def modal_proxy_auth_headers() -> dict[str, str]:
    """Return Modal proxy-auth headers from env or saved local credentials."""
    load_proxy_auth()
    key = os.environ.get("MODAL_KEY", "").strip()
    secret = os.environ.get("MODAL_SECRET", "").strip()
    if key and secret:
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


def get_framework_status_url() -> str | None:
    """Resolve the framework-status endpoint URL, or ``None``.

    The ``TRAINING_GYM_FRAMEWORK_STATUS_URL`` env var takes precedence when set,
    so callers on the driver and inside remote containers resolve the same
    endpoint; otherwise the URL is derived from the saved dashboard URL.
    """
    override = os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "").strip()
    if override:
        return override
    base = get_dashboard_url()
    if not base:
        return None
    return base.rstrip("/") + "/api/framework-status"


def _render(config: dict[str, Any]) -> str:
    """Minimal TOML writer for the shapes we persist (string-valued tables)."""
    lines: list[str] = []
    for section, entries in config.items():
        if not isinstance(entries, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in entries.items():
            lines.append(f"{key} = {_format_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


# ── Modal credential resolution ──────────────────────────────────────────


def read_modal_toml_creds() -> tuple[str, str, str]:
    """Resolve ``(token_id, token_secret, profile_name)`` from ``~/.modal.toml``."""
    if not MODAL_CONFIG_PATH.is_file():
        return "", "", ""

    try:
        with MODAL_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return "", "", ""

    profiles = {
        name: section for name, section in data.items() if isinstance(section, dict)
    }
    if not profiles:
        return "", "", ""

    candidate_names: list[str] = []
    env_profile = os.environ.get("MODAL_PROFILE", "").strip()
    if env_profile:
        candidate_names.append(env_profile)
    candidate_names.extend(
        name for name, sec in profiles.items() if sec.get("active") is True
    )
    if "default" in profiles:
        candidate_names.append("default")
    candidate_names.extend(profiles.keys())

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen or name not in profiles:
            continue
        seen.add(name)
        section = profiles[name]
        token_id = str(section.get("token_id") or "").strip()
        token_secret = str(section.get("token_secret") or "").strip()
        if token_id and token_secret:
            return token_id, token_secret, name
    return "", "", ""


def resolve_modal_creds() -> tuple[str, str, str]:
    """Resolve Modal credentials with a source label for logging.

    Order: ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` env vars → active
    profile in ``~/.modal.toml``.
    """
    env_id = (os.environ.get("MODAL_TOKEN_ID") or "").strip()
    env_secret = (os.environ.get("MODAL_TOKEN_SECRET") or "").strip()
    if env_id and env_secret:
        return env_id, env_secret, "environment"

    toml_id, toml_secret, profile_name = read_modal_toml_creds()
    if toml_id and toml_secret:
        return toml_id, toml_secret, f"~/.modal.toml profile [{profile_name}]"
    return "", "", ""
