"""Explicit, resumable migration of Training Gym resources."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass

from .errors import CLIError
from .output import print_note
from .setup import ProxyAuthMode, setup
from modal_dojo.common import config
from modal_dojo.common.dashboard import (
    DASHBOARD_APP_NAME,
    LEGACY_DASHBOARD_APP_NAME,
    deployed_dashboard_url,
)
from modal_dojo.common.dashboard_components import DASHBOARD_OVERLAY_VOLUME_NAME
from modal_dojo.common.trackio import (
    TrackioConfig,
    _DEFAULT_MODAL_APP_NAME,
)
from modal_dojo.utils.metadata import METADATA_VOLUME_NAME

VOLUME_PAIRS = (
    ("training-gym-metadata", METADATA_VOLUME_NAME),
    ("training-gym-dashboard-overlay", DASHBOARD_OVERLAY_VOLUME_NAME),
    ("training-gym-trackio-data", f"{_DEFAULT_MODAL_APP_NAME}-data"),
)


@dataclass
class AppInfo:
    id: str
    name: str
    live: bool | None
    tags: dict[str, str]


class ModalResources:
    """Modal operations isolated from migration sequencing for local tests."""

    def apps(self, names):
        import modal
        from modal.exception import NotFoundError

        result = []
        for name in sorted(names):
            try:
                app = modal.App.lookup(name, create_if_missing=False)
            except NotFoundError:
                continue
            result.append(AppInfo(app.app_id, name, True, {}))
        return result

    def volume(self, name):
        import modal
        from modal.exception import NotFoundError

        volume = modal.Volume.from_name(name)
        try:
            volume.hydrate()
        except NotFoundError:
            return None
        return volume

    def records(self, volume):
        """Read training statuses with one summary-file read, without listing."""
        from modal_dojo.utils.metadata import (
            MetadataStore,
            SUMMARY_KEY,
            SUMMARY_ITEMS_KEY,
        )

        path = f"{MetadataStore.TRAINING_RUNS_SUMMARY.value}/{SUMMARY_KEY}.json"
        try:
            payload = json.loads(b"".join(volume.read_file(path)))
            items = (
                payload.get(SUMMARY_ITEMS_KEY) if isinstance(payload, dict) else payload
            )
            if not isinstance(items, list) or any(
                not isinstance(item, dict) for item in items
            ):
                raise ValueError("expected a list of training-run summaries")
        except Exception as exc:
            raise ValueError(
                f"Cannot read training-run summary {path}. Manually check that no runs are active, then run with --force."
            ) from exc
        for index, item in enumerate(items):
            run_id = item.get("training_run_id", f"summary-row-{index}")
            yield f"training-runs/{run_id}", item

    def stop(self, app_id):
        """Stop via the public CLI and wait for its running containers to exit."""
        command = [sys.executable, "-m", "modal"]
        subprocess.run(
            [*command, "app", "stop", app_id, "--yes"], check=True, timeout=60
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = subprocess.run(
                [*command, "container", "list", "--app-id", app_id, "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            containers = json.loads(result.stdout)
            if not isinstance(containers, list):
                raise ValueError(f"Unexpected container list for {app_id}")
            if not containers:
                return
            time.sleep(1)
        raise ValueError(f"Could not verify that {app_id} stopped")

    def rename(self, old, new):
        import modal

        modal.Volume.rename(old, new)


def _check_runs(resources, volumes):
    terminal = {"completed", "failed", "cancelled", "canceled", "stopped"}
    blockers = []
    for volume in volumes:
        for path, record in resources.records(volume):
            if not path.lstrip("/").startswith("training-runs/"):
                continue
            status = record.get("status")
            if not isinstance(status, str) or status not in terminal:
                app_id = record.get("modal_app_id") or record.get("app_id")
                detail = f", app {app_id}" if app_id else ""
                blockers.append(f"{path} (status={status!r}{detail})")
    if blockers:
        raise ValueError(
            "Active or unknown run status in metadata prevents migration: "
            + ", ".join(blockers)
            + ". Finish these runs before retrying."
        )


def migrate(
    *, proxy_auth=ProxyAuthMode.UNSPECIFIED, force: bool = False, resources=None
):
    """Migrate from the old Training Gym to a new Modal Dojo configuration."""
    from .skills import warn_legacy_skills

    warn_legacy_skills()
    resources = resources or ModalResources()
    stage = "configuration inspection"
    print("Checking local configuration...")
    try:
        old_exists = config.LEGACY_CONFIG_PATH.exists()
        new_exists = config.CONFIG_PATH.exists()
        if not old_exists and not new_exists:
            raise ValueError("No configuration found. Run `modal-dojo setup`.")
        legacy = (
            config._read_config(config.LEGACY_CONFIG_PATH, strict=True)
            if old_exists
            else {}
        )
        settings = (
            config._read_config(config.CONFIG_PATH, strict=True)
            if new_exists
            else legacy
        )
        stage = "volume preflight"
        print("Checking old and new volume names for conflicts...")
        volumes = {}
        for old, new in VOLUME_PAIRS:
            source, target = resources.volume(old), resources.volume(new)
            if source is not None and target is not None:
                raise ValueError(
                    f"Both {old} and {new} exist. Resolve the conflict manually then retry; neither will be overwritten."
                )
            volumes[new] = source if source is not None else target
        trackio = volumes[f"{_DEFAULT_MODAL_APP_NAME}-data"] is not None
        services = {LEGACY_DASHBOARD_APP_NAME, DASHBOARD_APP_NAME}
        if trackio:
            services.update({_DEFAULT_MODAL_APP_NAME, "training-gym-trackio"})
        metadata = (
            [volumes[METADATA_VOLUME_NAME]]
            if volumes[METADATA_VOLUME_NAME] is not None
            else []
        )
        stage = "live-run check"
        if force:
            print("Skipping the live-run check (--force).")
        else:
            print("Checking for active runs...")
            _check_runs(resources, metadata)
        stage = "service discovery"
        apps = resources.apps(services)
        stage = "authentication resolution"
        print("Resolving dashboard authentication settings...")
        url = (
            deployed_dashboard_url()
            or deployed_dashboard_url(LEGACY_DASHBOARD_APP_NAME)
            or settings.get("dashboard", {}).get("url")
        )
        tokens = settings.get("proxy_auth", {})
        key = os.environ.get("MODAL_KEY") or tokens.get("key")
        secret = os.environ.get("MODAL_SECRET") or tokens.get("secret")
        headers = {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}
        if proxy_auth is ProxyAuthMode.UNSPECIFIED:
            mode = config.get_dashboard_proxy_auth(
                url, settings=settings, headers=headers
            )
            if mode is None:
                raise ValueError(
                    "Cannot determine auth mode. Pass --proxy-auth or --no-proxy-auth to modal-dojo migrate."
                )
            proxy_auth = ProxyAuthMode.REQUIRE if mode else ProxyAuthMode.DISABLE
        stage = "service shutdown"
        print("Stopping dashboard services and waiting for their containers to exit...")
        for app in apps:
            if app.name in services and app.live is not False:
                print(f"Stopping {app.name} ({app.id})...")
                resources.stop(app.id)
                print(f"Stopped {app.name}")
        stage = "volume rename"
        print("Renaming volumes...")
        for old, new in VOLUME_PAIRS:
            source = resources.volume(old)
            if source is not None:
                if resources.volume(new) is not None:
                    raise ValueError(
                        f"{new} appeared during migration; resolve manually"
                    )
                identity = source.object_id
                print(f"Renaming {old} to {new}...")
                resources.rename(old, new)
                if resources.volume(new).object_id != identity:
                    raise ValueError(f"Volume identity changed for {new}")
                print_note(f"Renamed {old} to {new}")
        stage = "configuration move"
        print("Saving configuration at the Modal Dojo path...")
        if old_exists and not new_exists:
            if config.CONFIG_PATH.exists():
                raise ValueError(
                    "New config appeared during migration; rerun to inspect it"
                )
            config.LEGACY_CONFIG_PATH.rename(config.CONFIG_PATH)
            config.CONFIG_PATH.chmod(0o600)
        # Store the resolved choice even if a subsequent deployment fails.
        settings.setdefault("dashboard", {})["proxy_auth"] = (
            proxy_auth is ProxyAuthMode.REQUIRE
        )
        config._write_config(config._render(settings).encode())
        stage = "Trackio deployment"
        if trackio:
            print("Deploying Trackio with the migrated data volume...")
            result = TrackioConfig.deploy_to_modal()
            print_note(f"Trackio deployed: {result.server_url}")
        else:
            print("No default Trackio data volume found; skipping Trackio deployment.")
        stage = "dashboard deployment"
        print("Deploying the Modal Dojo dashboard...")
        url = setup(proxy_auth=proxy_auth)
        if old_exists and new_exists:
            backup = config.LEGACY_CONFIG_PATH.with_name(
                config.LEGACY_CONFIG_PATH.name + "." + uuid.uuid4().hex + ".bak"
            )
            config.LEGACY_CONFIG_PATH.rename(backup)
            backup.chmod(0o600)
            print_note(f"Archived legacy configuration at {backup}")
        print("Migration complete!")
        return url
    except Exception as exc:
        raise CLIError(
            f"Migration failed during {stage}: {exc}",
            error="migration_failed",
            hint=(
                "After verifying there are no active training runs, rerun `modal-dojo migrate --force` to skip the live-run check."
                if stage == "live-run check"
                else "Resolve the issue and rerun modal-dojo migrate. Completed renames are retained."
            ),
        ) from exc
