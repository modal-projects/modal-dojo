"""Identity of the deployed modal-dojo dashboard app.

Lives in ``common`` so library code — ``common.config``, the launchers — can
resolve the live dashboard without importing ``cli``, which imports ``common``.
"""

from __future__ import annotations

import importlib.metadata
import os
import tomllib
from pathlib import Path

DASHBOARD_APP_NAME = "dojo-dashboard"
LEGACY_DASHBOARD_APP_NAME = "training-gym-dashboard"
DASHBOARD_WEB_FUNCTION = "fastapi_app"

# Set on a per-PR dashboard deploy (scripts/previews/dashboard_api.py) so it
# serves the API without the scheduled jobs that rewrite shared metadata.
DASHBOARD_PREVIEW_ENV_KEY = "MODAL_DOJO_DASHBOARD_PREVIEW"
DASHBOARD_VERSION_ENV_KEY = "DASHBOARD_VERSION"


def current_dashboard_version() -> str:
    if baked := os.environ.get(DASHBOARD_VERSION_ENV_KEY):
        return baked
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    return importlib.metadata.version("modal-dojo")


def is_dashboard_upgrade(incoming: str, deployed: str | None) -> bool:
    if not deployed:
        return True
    if incoming.isdigit() != deployed.isdigit():
        return deployed.isdigit()
    from packaging.version import InvalidVersion, Version

    try:
        return Version(incoming) > Version(deployed)
    except InvalidVersion:
        return True


class DashboardLookupUnknown(Exception):
    """Modal lookup of the dashboard app failed before not-found could be observed."""


def deployed_dashboard_url(
    app_name: str = DASHBOARD_APP_NAME, *, environment_name: str | None = None
) -> str | None:
    """Return the live dashboard web URL if its app is deployed, else ``None``.

    Raises ``DashboardLookupUnknown`` when lookup fails before that can
    be observed (timeout, auth, network, unexpected Modal errors).
    """
    import modal
    from modal.exception import NotFoundError

    try:
        fn = modal.Function.from_name(
            app_name, DASHBOARD_WEB_FUNCTION, environment_name=environment_name
        )
        fn.hydrate()
        url = fn.get_web_url()
    except NotFoundError:
        return None
    except Exception as exc:
        raise DashboardLookupUnknown from exc
    if not url:
        raise DashboardLookupUnknown
    return url
