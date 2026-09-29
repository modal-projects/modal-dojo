"""Deploy the modal-dojo dashboard to Modal.

Usage (CLI):
    modal-dojo setup

What this does:
1. Provisions a ``_training-gym-modal-creds`` Modal Secret containing
   ``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET``. The deployed dashboard uses
   these credentials to stream logs from *other* Modal apps (the user's
   training runs) into the dashboard UI — container-default creds aren't
   scoped to read cross-app logs, so a workspace token is required.

   The underscore prefix marks the secret as auto-managed (hidden from the
   normal Secrets list in the UI). Credentials are auto-resolved from
   ``MODAL_TOKEN_*`` env vars or the active profile in ``~/.modal.toml``.

2. Deploys the dashboard's ASGI app and orphan reconciler.
3. Persists the FastAPI web URL to ``~/.modal-dojo.toml`` so other
   clients (e.g. the slime launcher) can find the dashboard.
"""

from __future__ import annotations

import os
from enum import Enum
import webbrowser
from pathlib import Path

from modal_dojo.common.dashboard import (
    DASHBOARD_APP_NAME,
    LEGACY_DASHBOARD_APP_NAME,
    DashboardLookupUnknown,
    current_dashboard_version,
    deployed_dashboard_url,
    is_dashboard_upgrade,
)


class ProxyAuthMode(Enum):
    REQUIRE = "require"
    DISABLE = "disable"
    UNSPECIFIED = "unspecified"


def _load_dashboard_for_deploy(
    requires_proxy_auth: bool, trajectory_viewer: str | None = None
):
    """Import the declarative dashboard with the selected ASGI proxy-auth mode."""
    import importlib
    import sys

    from modal_dojo.common import config

    module_name = "modal_dojo._dashboard"
    config.set_dashboard_requires_proxy_auth(requires_proxy_auth)
    config.set_dashboard_trajectory_viewer(trajectory_viewer)
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def setup(
    proxy_auth: ProxyAuthMode = ProxyAuthMode.UNSPECIFIED,
    interactive: bool = True,
    trajectory_viewer: str | Path | None = None,
    reset_trajectory_viewer: bool = False,
) -> str:
    """Deploy the modal-dojo dashboard, persist its URL, and return it.

    ``interactive=False`` resolves Modal credentials silently (from env vars
    or ``~/.modal.toml``) and never prompts — used by the auto-deploy path in
    ``TrainConfig.train()`` so a training run is never blocked on stdin.
    """
    import modal

    from modal_dojo.common.config import (
        CONFIG_PATH,
        get_dashboard_trajectory_viewer,
        migrate_config,
        get_dashboard_url,
        get_dashboard_proxy_auth,
        save_dashboard_trajectory_viewer,
        save_dashboard_url,
    )

    from modal_dojo.cli.output import print_note

    from .skills import warn_legacy_skills

    warn_legacy_skills()
    copied = migrate_config()
    saved_url = get_dashboard_url()
    try:
        new_url = deployed_dashboard_url()
        old_url = deployed_dashboard_url(LEGACY_DASHBOARD_APP_NAME)
    except DashboardLookupUnknown as exc:
        raise ValueError(
            "Could not discover existing dashboards. Check Modal credentials/network and rerun modal-dojo setup."
        ) from exc
    migrating = (
        copied
        or bool(old_url)
        or bool(saved_url and "--training-gym-dashboard-" in saved_url)
    )
    existing_url = new_url or old_url or saved_url
    if proxy_auth is ProxyAuthMode.UNSPECIFIED:
        inherited = get_dashboard_proxy_auth(existing_url)
        if inherited is None and (migrating or existing_url):
            raise ValueError(
                "Cannot determine existing dashboard authentication. Pass --proxy-auth or --no-proxy-auth explicitly to modal-dojo setup."
            )
        require_proxy_auth = inherited is True
    elif proxy_auth is ProxyAuthMode.REQUIRE:
        require_proxy_auth = True
    elif proxy_auth is ProxyAuthMode.DISABLE:
        require_proxy_auth = False
    else:
        raise TypeError("proxy_auth must be a ProxyAuthMode")

    if reset_trajectory_viewer:
        save_dashboard_trajectory_viewer(None)
        trajectory_viewer_path = None
    elif trajectory_viewer is not None:
        viewer_path = Path(trajectory_viewer).expanduser().resolve()
        if not viewer_path.is_file():
            raise ValueError(f"trajectory viewer must be a file: {viewer_path}")
        trajectory_viewer_path = str(viewer_path)
        save_dashboard_trajectory_viewer(trajectory_viewer_path)
    else:
        trajectory_viewer_path = get_dashboard_trajectory_viewer()

    if require_proxy_auth:
        print("Deploying dashboard with proxy authentication enabled.")
    else:
        print("This dashboard will not have proxy authentication enabled.")
        print("If you would like to enable it, run `modal-dojo setup --proxy-auth`.")
        print()

    if trajectory_viewer_path:
        print(f"Using custom trajectory viewer: {trajectory_viewer_path}")
    dashboard = _load_dashboard_for_deploy(
        require_proxy_auth, trajectory_viewer=trajectory_viewer_path
    )

    has_proxy_auth_token = ensure_proxy_auth(interactive=interactive)
    if require_proxy_auth and not has_proxy_auth_token:
        print(
            "WARNING: Dashboard proxy auth requires MODAL_KEY and MODAL_SECRET. "
            "Run `modal-dojo set-proxy-auth` or export both variables."
        )

    if not dashboard.ensure_creds_secret(interactive=interactive):
        print(
            f"WARNING: continuing without the {dashboard.MODAL_CREDS_SECRET_NAME!r} "
            "Modal Secret — the dashboard will not be able to stream Modal "
            "app logs."
        )

    with modal.enable_output():
        dashboard.app.deploy()

    web_url = dashboard.fastapi_app.get_web_url()
    if not web_url:
        raise ValueError(
            "Dashboard deployment returned no URL; previous saved URL retained."
        )
    save_dashboard_url(web_url, proxy_auth=require_proxy_auth)
    if migrating:
        print_note(
            "A new modal-dojo-dashboard has been deployed. The old training-gym-dashboard app has not been stopped. After active runs finish, run `modal app stop training-gym-dashboard` in the same Modal environment, then update saved dashboard links to the new URL."
        )
    print(f"\nDashboard deployed: {web_url}")
    print(f"Saved dashboard URL to {CONFIG_PATH}")
    return web_url


def ensure_proxy_auth(interactive: bool = True, force: bool = False) -> bool:
    """Prompt for and persist Modal proxy-auth tokens in ``~/.modal-dojo.toml``.

    Authenticated served endpoints (``unauthenticated=False``) need Modal proxy auth
    and need a ``MODAL_KEY`` / ``MODAL_SECRET`` token pair. When ``interactive``
    we ask whether the user has created a pair and, if so, read and save it.
    With ``force`` we offer to replace an already-saved pair (e.g. when the old
    one was minted in the wrong workspace). Returns ``True`` when both tokens are
    available afterwards.
    """
    from getpass import getpass

    from modal_dojo.common.config import (
        CONFIG_PATH,
        get_proxy_auth,
        load_proxy_auth,
        save_proxy_auth,
    )

    key, secret = get_proxy_auth()
    if not force:
        load_proxy_auth()
        key = os.environ.get("MODAL_KEY", "").strip() or key
        secret = os.environ.get("MODAL_SECRET", "").strip() or secret
    if key and secret and not force:
        return True
    if not interactive:
        return bool(key and secret)

    if key and secret:
        print(
            f"\nA proxy-auth pair is already saved in {CONFIG_PATH} "
            f"(MODAL_KEY {key[:6]}…). Proxy-auth tokens are workspace-scoped — "
            "if endpoints return 401 the pair was likely minted in the wrong "
            "workspace."
        )
        answer = input("Replace it? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return True
    else:
        print(
            "\nAuthenticated custom deployments "
            "(CustomDeployment.launch(..., unauthenticated=False)) need "
            "a Modal proxy-auth token pair (MODAL_KEY / MODAL_SECRET)."
        )
        answer = (
            input(
                "Have you created a proxy-auth token pair at "
                "https://modal.com/settings/proxy-auth-tokens? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            print(
                "Skipping proxy-auth setup. Create a pair at "
                "https://modal.com/settings/proxy-auth-tokens and re-run "
                "`modal-dojo set-proxy-auth`, or export MODAL_KEY / "
                "MODAL_SECRET yourself."
            )
            return False

    key = input("MODAL_KEY (wk-…): ").strip()
    secret = getpass("MODAL_SECRET (ws-…): ").strip()
    if not (key and secret):
        print("Both MODAL_KEY and MODAL_SECRET are required — skipping.")
        return False
    if not key.startswith("wk-") or not secret.startswith("ws-"):
        print(
            "WARNING: expected MODAL_KEY to start with 'wk-' and MODAL_SECRET "
            "with 'ws-'. Saving anyway."
        )

    save_proxy_auth(key, secret)
    print(f"Saved proxy-auth tokens to {CONFIG_PATH}")
    print(
        "\nThis token cannot access any environments by default. You must "
        "configure which environments it is valid for before it can be used.\n"
        "Set this at https://modal.com/settings/proxy-auth-tokens — make sure "
        "the environment your endpoint runs in is enabled for "
        "this token."
    )
    return True


def set_proxy_auth() -> bool:
    """Interactively (re)set the saved proxy-auth token pair.

    Thin wrapper over :func:`ensure_proxy_auth` with ``force=True`` so an
    existing pair is replaced — exposed as ``modal-dojo set-proxy-auth``.
    """
    return ensure_proxy_auth(interactive=True, force=True)


def set_password(password: str | None = None) -> None:
    """Set or clear the dashboard password, then redeploy so it takes effect.

    Pass an empty string to disable auth. When ``password`` is ``None`` we
    prompt for it (hidden input). The deployed app reads the value from its
    environment at startup, so we redeploy after updating the Secret.
    """
    from getpass import getpass

    from modal_dojo._dashboard import set_dashboard_password

    if password is None:
        password = getpass("Dashboard password (leave empty to disable auth): ").strip()
        if password:
            confirm = getpass("Confirm password: ").strip()
            if confirm != password:
                print("Passwords don't match — aborting.")
                return

    set_dashboard_password(password)
    if password:
        print("Dashboard password set. Redeploying so it takes effect...")
    else:
        print("Dashboard password cleared (open access). Redeploying...")

    from modal_dojo.cli.output import print_warning
    from modal_dojo.common.trackio import (
        TrackioLookupUnknown,
        lookup_trackio_url,
    )

    setup(
        interactive=False,
        proxy_auth=ProxyAuthMode.UNSPECIFIED,
    )
    try:
        trackio_url = lookup_trackio_url()
    except TrackioLookupUnknown as exc:
        print_warning(
            f"Could not check whether a Trackio dashboard is deployed "
            f"({exc.__cause__ or exc}). Re-run TrackioConfig.deploy_to_modal() "
            "if you have one so the new password takes effect."
        )
        return
    if trackio_url:
        print_warning(
            "A Trackio dashboard is deployed on Modal and still has the old password. "
            "Re-run TrackioConfig.deploy_to_modal() so the new password takes effect."
        )


def open_dashboard() -> str | None:
    """Open the deployed dashboard in the default browser; return its URL.

    Resolves the live URL from Modal (authoritative) and keeps the cached
    ``~/.modal-dojo.toml`` value in sync, falling back to that cache if the
    Modal lookup fails. Prints guidance and returns ``None`` when nothing is
    deployed.
    """
    from modal_dojo.common.config import get_dashboard_url, save_dashboard_url

    try:
        web_url = deployed_dashboard_url()
    except DashboardLookupUnknown:
        web_url = None
    if web_url:
        if get_dashboard_url() != web_url:
            save_dashboard_url(web_url)
    else:
        web_url = get_dashboard_url()

    if not web_url:
        print(
            "No deployed modal-dojo dashboard found. "
            "Run `modal-dojo setup` to deploy it first."
        )
        return None

    print(f"Opening dashboard: {web_url}")
    webbrowser.open(web_url)
    return web_url


def ensure_dashboard_deployed() -> str | None:
    """Best-effort dashboard provision for ``train()`` and ``evaluate()``.

    Never raises; failures warn so the run continues.
    """
    try:
        from modal_dojo.common.config import (
            DashboardVersionUnknown,
            get_dashboard_url,
            get_dashboard_version,
            save_dashboard_url,
        )

        try:
            web_url = deployed_dashboard_url()
        except DashboardLookupUnknown:
            return get_dashboard_url()
        incoming = current_dashboard_version()
        if web_url:
            if get_dashboard_url() != web_url:
                save_dashboard_url(web_url)
            try:
                deployed = get_dashboard_version(web_url)
            except DashboardVersionUnknown:
                return web_url
            if not is_dashboard_upgrade(incoming, deployed):
                return web_url
            print(
                f"modal-dojo dashboard ({DASHBOARD_APP_NAME!r}) is older than "
                f"this dashboard ({incoming}): redeploying it."
            )
        else:
            print(
                f"modal-dojo dashboard ({DASHBOARD_APP_NAME!r}) is not deployed: "
                "deploying it now."
            )
        return setup(
            interactive=False,
            proxy_auth=ProxyAuthMode.UNSPECIFIED,
        )
    except Exception as exc:
        print(
            f"WARNING: could not ensure the modal-dojo dashboard is deployed: "
            f"{exc}. Continuing without dashboard status reporting; run "
            "`modal-dojo setup` to deploy it manually."
        )
        return None
