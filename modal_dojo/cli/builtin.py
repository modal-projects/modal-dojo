"""Click wrappers for the CLI's existing top-level commands."""

from __future__ import annotations

from pathlib import Path

import click

from .commands import _DojoCommand


@click.command("setup", cls=_DojoCommand)
@click.option(
    "--proxy-auth",
    is_flag=True,
    help="Require Modal proxy authentication for the dashboard.",
)
@click.option(
    "--no-proxy-auth",
    is_flag=True,
    help="Deploy the dashboard without Modal proxy authentication.",
)
@click.option(
    "--trajectory-viewer",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Mount a custom TrajectoryViewer.svelte over the dashboard default.",
)
@click.option(
    "--no-trajectory-viewer",
    is_flag=True,
    help="Forget a saved --trajectory-viewer and redeploy the built-in viewer.",
)
def setup_command(
    proxy_auth: bool,
    no_proxy_auth: bool,
    trajectory_viewer: Path | None,
    no_trajectory_viewer: bool,
) -> None:
    """Deploy the dashboard."""
    if proxy_auth and no_proxy_auth:
        raise click.UsageError(
            "--proxy-auth and --no-proxy-auth cannot be used together."
        )
    if trajectory_viewer is not None and no_trajectory_viewer:
        raise click.UsageError(
            "--trajectory-viewer and --no-trajectory-viewer cannot be used together."
        )

    from .setup import ProxyAuthMode, setup

    selected_auth = ProxyAuthMode.UNSPECIFIED
    if proxy_auth:
        selected_auth = ProxyAuthMode.REQUIRE
    elif no_proxy_auth:
        selected_auth = ProxyAuthMode.DISABLE

    if no_trajectory_viewer:
        setup(proxy_auth=selected_auth, reset_trajectory_viewer=True)
    elif trajectory_viewer is None:
        setup(proxy_auth=selected_auth)
    else:
        setup(proxy_auth=selected_auth, trajectory_viewer=trajectory_viewer)


@click.command("open", cls=_DojoCommand)
def open_command() -> None:
    """Open the dashboard."""
    from .setup import open_dashboard

    open_dashboard()


@click.command("set-proxy-auth", cls=_DojoCommand)
def set_proxy_auth_command() -> None:
    """Store Modal proxy-auth credentials.

    Used by authenticated custom deployments when `MODAL_KEY` and
    `MODAL_SECRET` are not set in the environment. Re-running replaces the saved credentials.
    """
    from .setup import set_proxy_auth

    set_proxy_auth()


@click.command("set-password", cls=_DojoCommand)
@click.option(
    "--password",
    default=None,
    metavar="PASSWORD",
    help="Set a password, prompt when omitted, or disable authentication with an empty value.",
)
def set_password_command(password: str | None) -> None:
    """Set or clear dashboard Basic Auth, then redeploy."""
    from .setup import set_password

    set_password(password=password)


@click.command("cleanup", cls=_DojoCommand)
@click.option(
    "--older-than-days",
    type=int,
    default=7,
    show_default=True,
    metavar="DAYS",
    help="Delete failed or cancelled runs older than this many days.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting.",
)
def cleanup_command(older_than_days: int, dry_run: bool) -> None:
    """Delete old failed or cancelled run metadata."""
    from .cleanup import cleanup

    cleanup(older_than_days=older_than_days, dry_run=dry_run)


@click.command("migrate", cls=_DojoCommand)
@click.option(
    "--proxy-auth", is_flag=True, help="Require dashboard proxy authentication."
)
@click.option(
    "--no-proxy-auth", is_flag=True, help="Disable dashboard proxy authentication."
)
@click.option(
    "--force",
    is_flag=True,
    help="Skip the live-run check after verifying no training runs are active.",
)
def migrate_command(proxy_auth: bool, no_proxy_auth: bool, force: bool) -> None:
    """Migrate Training Gym configuration and volumes during a maintenance window."""
    from .migrate import migrate
    from .setup import ProxyAuthMode

    if proxy_auth and no_proxy_auth:
        raise click.UsageError(
            "--proxy-auth and --no-proxy-auth cannot be used together."
        )
    mode = (
        ProxyAuthMode.REQUIRE
        if proxy_auth
        else ProxyAuthMode.DISABLE
        if no_proxy_auth
        else ProxyAuthMode.UNSPECIFIED
    )
    migrate(proxy_auth=mode, force=force)
