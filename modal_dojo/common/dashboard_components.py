"""Run-scoped dashboard component artifacts.

Components are stored content-addressably so a dashboard component can be
attached to a run after launch without changing the training configuration.
The manifest in the run record is the stable hand-off used by dashboard
overlays to resolve the component source.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any


DASHBOARD_OVERLAY_VOLUME_NAME = "modal-dojo-dashboard-overlay"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_COMPONENT_BYTES = 2 * 1024 * 1024
# Size budget for one compiled component (its Svelte runtime and everything it
# imports, bundled into the frame). A larger bundle fails to compile.
MAX_COMPILED_COMPONENT_BYTES = 400_000


class DashboardComponent(str, Enum):
    """Supported run-scoped dashboard component slots.

    Every slot takes a Svelte component that the dashboard compiles and runs in
    a sandboxed ``<iframe>``. The props each slot receives are listed in
    :data:`COMPONENT_PROPS`.

    - ``TRAJECTORY_VIEWER`` replaces the conversation view of the selected
      sample. Only the most recently attached one is shown.
    - ``SUMMARY_PANEL`` adds a panel to the run's Summary tab.
    - ``ROLLOUT_PANEL`` adds a panel under the selected sample on the
      Rollouts tab.

    A run can attach several panels of each panel type; they render in
    attachment order.
    """

    TRAJECTORY_VIEWER = "trajectory_viewer"
    SUMMARY_PANEL = "summary_panel"
    ROLLOUT_PANEL = "rollout_panel"


# The props each slot's component receives, as the dashboard posts them. Keep
# this in step with the frontend call sites and dashboards/README.md.
_SAMPLE_PROPS = {
    "sample": "the selected normalized sample",
    "samples": "all samples in the selected rollout group",
    "rollout": "the expanded TrainingRolloutResult",
    "run": "the TrainingRun summary",
    "position": "the selected sample's position, as '<bucket>/<index>'",
}
COMPONENT_PROPS: dict[DashboardComponent, dict[str, str]] = {
    DashboardComponent.TRAJECTORY_VIEWER: {
        **_SAMPLE_PROPS,
        "trajectory": "sample.metadata.trajectory_messages",
        "rewardEvents": "sample.reward_events",
    },
    DashboardComponent.SUMMARY_PANEL: {
        "run": "the TrainingRun summary",
        "rollouts": "per-step rollout summaries, oldest first",
    },
    DashboardComponent.ROLLOUT_PANEL: dict(_SAMPLE_PROPS),
}

# Slots that show every attached component rather than only the latest one.
MULTI_INSTANCE_COMPONENTS = frozenset(
    {DashboardComponent.SUMMARY_PANEL, DashboardComponent.ROLLOUT_PANEL}
)


def _coerce_component_type(
    component_type: DashboardComponent | str,
) -> DashboardComponent:
    try:
        return DashboardComponent(component_type)
    except ValueError as exc:
        supported = ", ".join(item.value for item in DashboardComponent)
        raise ValueError(
            f"unsupported dashboard component type {component_type!r}; "
            f"supported types: {supported}"
        ) from exc


def read_dashboard_component(
    *,
    component_type: DashboardComponent | str,
    from_path: str | Path,
) -> tuple[Path, bytes, str]:
    """Validate a local component file and return ``(path, data, sha256)``."""
    component_type = _coerce_component_type(component_type)
    path = Path(from_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"dashboard component file does not exist: {path}")
    if path.suffix != ".svelte":
        raise ValueError(
            f"{component_type.value} components must be Svelte files (.svelte)"
        )

    data = path.read_bytes()
    if not data:
        raise ValueError(f"dashboard component file is empty: {path}")
    if len(data) > MAX_COMPONENT_BYTES:
        raise ValueError(
            f"dashboard component file is too large ({len(data)} bytes; "
            f"maximum {MAX_COMPONENT_BYTES} bytes): {path}"
        )

    return path, data, hashlib.sha256(data).hexdigest()


def store_dashboard_component(
    *,
    name: str,
    component_type: DashboardComponent | str,
    from_path: str | Path,
    training_run_id: str | None = None,
) -> dict[str, Any]:
    """Upload a local component and return its immutable artifact manifest.

    When ``training_run_id`` is provided, a second association manifest is
    written under ``runs/<training_run_id>/<name>.json`` so a mounted dashboard
    can resolve the registration without relying on a local filesystem.
    """
    component_type = _coerce_component_type(component_type)
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(
            "dashboard component name must be 1-64 characters containing only "
            "letters, numbers, '.', '_' or '-'."
        )

    path, data, digest = read_dashboard_component(
        component_type=component_type, from_path=from_path
    )
    remote_path = f"components/{component_type.value}/{digest}/{name}{path.suffix}"
    manifest_path = f"components/{component_type.value}/{digest}/manifest.json"
    manifest = {
        "name": name,
        "type": component_type.value,
        "sha256": digest,
        "filename": path.name,
        "size": len(data),
        "path": remote_path,
        "contract": "modal-dojo-dashboard-component/v1",
    }
    if training_run_id:
        manifest["training_run_id"] = training_run_id
    association_path = (
        f"runs/{training_run_id}/{name}.json" if training_run_id else None
    )

    # Import Modal lazily so importing TrainingRun remains cheap and local
    # metadata operations do not require a client until an upload is requested.
    import modal

    volume = modal.Volume.from_name(
        DASHBOARD_OVERLAY_VOLUME_NAME, create_if_missing=True
    )
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(path), remote_path)
        batch.put_file(
            BytesIO(json.dumps(manifest, separators=(",", ":")).encode()),
            manifest_path,
        )
        if association_path:
            batch.put_file(
                BytesIO(json.dumps(manifest, separators=(",", ":")).encode()),
                association_path,
            )
    # ``batch_upload`` commits its batch when the context exits.  Calling
    # ``Volume.commit`` here would only work for a volume mounted inside a
    # Modal container and fails for the local client used by this API.
    return manifest


def remove_dashboard_component_association(*, training_run_id: str, name: str) -> None:
    """Delete a run's association record for ``name`` from the overlay Volume.

    The content-addressed source stays: another run (or another name) can
    point at the same digest, and the dashboard only serves sources that a
    run's metadata still references. A missing record is not an error.
    """
    import modal
    from modal.exception import NotFoundError

    volume = modal.Volume.from_name(
        DASHBOARD_OVERLAY_VOLUME_NAME, create_if_missing=True
    )
    try:
        volume.remove_file(f"runs/{training_run_id}/{name}.json")
    except (FileNotFoundError, NotFoundError):
        pass


__all__ = [
    "COMPONENT_PROPS",
    "DASHBOARD_OVERLAY_VOLUME_NAME",
    "MULTI_INSTANCE_COMPONENTS",
    "MAX_COMPILED_COMPONENT_BYTES",
    "MAX_COMPONENT_BYTES",
    "DashboardComponent",
    "read_dashboard_component",
    "remove_dashboard_component_association",
    "store_dashboard_component",
]
