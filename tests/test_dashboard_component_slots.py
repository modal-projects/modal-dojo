from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo.common import dashboard_components as components_mod
from modal_dojo.common.dashboard_components import (
    COMPONENT_PROPS,
    MULTI_INSTANCE_COMPONENTS,
    DashboardComponent,
    read_dashboard_component,
)
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun

FRONTEND = Path(__file__).resolve().parents[1] / "dashboards" / "frontend"


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _manifest(name: str, kind: str, digest: str) -> dict[str, str]:
    return {
        "name": name,
        "type": kind,
        "sha256": digest,
        "path": f"components/{kind}/{digest}/{name}.svelte",
        "contract": "modal-dojo-dashboard-component/v1",
    }


def _attach(monkeypatch, run: TrainingRun, name: str, kind: str, digest: str):
    manifest = _manifest(name, kind, digest)
    monkeypatch.setattr(
        components_mod, "store_dashboard_component", lambda **_: manifest
    )
    run.add_dashboard_component(
        name=name, component_type=kind, from_path=name, replace=True
    )
    return manifest


@pytest.fixture
def no_association_cleanup(monkeypatch):
    removed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        components_mod,
        "remove_dashboard_component_association",
        lambda *, training_run_id, name: removed.append((training_run_id, name)),
    )
    return removed


def _new_run(run_id: str = "r") -> TrainingRun:
    TrainingRun(training_run_id=run_id, framework=Framework.SLIME, config={}).save()
    return TrainingRun.from_id(run_id)


def test_every_slot_documents_its_props() -> None:
    assert set(COMPONENT_PROPS) == set(DashboardComponent)
    assert MULTI_INSTANCE_COMPONENTS < set(DashboardComponent)
    assert DashboardComponent.TRAJECTORY_VIEWER not in MULTI_INSTANCE_COMPONENTS


@pytest.mark.parametrize("kind", list(DashboardComponent))
def test_every_slot_requires_svelte(kind, tmp_path) -> None:
    source = tmp_path / "Panel.svelte"
    source.write_text("<p>hi</p>")
    _, data, digest = read_dashboard_component(component_type=kind, from_path=source)
    assert digest == hashlib.sha256(data).hexdigest()

    script = tmp_path / "panel.js"
    script.write_text("export default 1")
    with pytest.raises(ValueError, match=r"must be Svelte files"):
        read_dashboard_component(component_type=kind, from_path=script)


def test_remove_detaches_one_component(
    monkeypatch, fake_volume, no_association_cleanup
) -> None:
    run = _new_run()
    _attach(monkeypatch, run, "a", "summary_panel", "a" * 64)
    _attach(monkeypatch, run, "b", "rollout_panel", "b" * 64)

    assert run.remove_dashboard_component("a") is True
    assert run.remove_dashboard_component("missing") is False

    stored = TrainingRun.from_id("r").metadata["dashboard_components"]
    assert list(stored) == ["b"]
    assert no_association_cleanup == [("r", "a")]


def test_stale_handle_cannot_resurrect_a_removed_component(
    monkeypatch, fake_volume, no_association_cleanup
) -> None:
    original = _new_run()
    _attach(monkeypatch, original, "panel", "summary_panel", "a" * 64)

    remover = TrainingRun.from_id("r")
    remover.remove_dashboard_component("panel")

    # `original` still carries "panel" in its copy of the metadata.
    original.save()
    assert TrainingRun.from_id("r").metadata["dashboard_components"] == {}


def test_removal_survives_a_concurrent_attachment(
    monkeypatch, fake_volume, no_association_cleanup
) -> None:
    first = _new_run()
    _attach(monkeypatch, first, "a", "summary_panel", "a" * 64)

    attacher = TrainingRun.from_id("r")
    remover = TrainingRun.from_id("r")
    _attach(monkeypatch, attacher, "b", "summary_panel", "b" * 64)
    remover.remove_dashboard_component("a")

    stored = TrainingRun.from_id("r").metadata["dashboard_components"]
    assert list(stored) == ["b"]


def _attach_all(run: TrainingRun) -> None:
    run.metadata["dashboard_components"] = {
        "viewer": _manifest("viewer", "trajectory_viewer", "a" * 64),
        "loss": _manifest("loss", "summary_panel", "b" * 64),
        "trace": _manifest("trace", "rollout_panel", "c" * 64),
        "reward": _manifest("reward", "summary_panel", "d" * 64),
        "broken": {"name": "broken", "type": "summary_panel"},
    }
    run.save()


def test_list_route_returns_attachments_in_order_with_frame_urls(
    fake_volume, monkeypatch, tmp_path
) -> None:
    _attach_all(_new_run("run-1"))
    with _client(monkeypatch, tmp_path) as client:
        everything = client.get("/api/runs/run-1/dashboard-components").json()
        panels = client.get(
            "/api/runs/run-1/dashboard-components", params={"type": "summary_panel"}
        ).json()
        unknown = client.get(
            "/api/runs/run-1/dashboard-components", params={"type": "nope"}
        )

    assert [item["name"] for item in everything] == [
        "viewer",
        "loss",
        "trace",
        "reward",
    ]
    assert [item["name"] for item in panels] == ["loss", "reward"]
    assert unknown.status_code == 404

    url = urlparse(panels[0]["frame_url"])
    assert url.path == (
        f"/api/runs/run-1/dashboard-components/summary_panel/{'b' * 64}/frame.html"
    )
    assert parse_qs(url.query) == {
        "bridge": ["modal-dojo-v1"],
        "v": [_dashboard.component_frame_version()],
    }


def test_single_route_serves_only_single_instance_slots(
    fake_volume, monkeypatch, tmp_path
) -> None:
    _attach_all(_new_run("run-1"))
    with _client(monkeypatch, tmp_path) as client:
        viewer = client.get("/api/runs/run-1/dashboard-components/trajectory_viewer")
        panel = client.get("/api/runs/run-1/dashboard-components/summary_panel")

    assert viewer.status_code == 200
    assert viewer.json()["sha256"] == "a" * 64
    assert viewer.json()["frame_url"].startswith(
        f"/api/runs/run-1/dashboard-components/trajectory_viewer/{'a' * 64}/frame.html?"
    )
    assert panel.status_code == 404
    assert "?type=" in panel.json()["detail"]


def test_frame_version_is_stable_and_short() -> None:
    version = _dashboard.component_frame_version()
    assert version == _dashboard.component_frame_version()
    assert len(version) == 16


@pytest.mark.skipif(
    shutil.which("node") is None or not (FRONTEND / "node_modules" / "svelte").is_dir(),
    reason="needs node and dashboards/frontend/node_modules (npm ci)",
)
def test_panel_frame_compiles_and_is_sandboxed(fake_volume, monkeypatch, tmp_path):
    source = b"<script>let { rollouts = [] } = $props();</script><p>{rollouts.length} steps</p>"
    digest = hashlib.sha256(source).hexdigest()
    overlay = tmp_path / "overlay"
    relative = f"components/summary_panel/{digest}/steps.svelte"
    (overlay / relative).parent.mkdir(parents=True)
    (overlay / relative).write_bytes(source)
    monkeypatch.setattr(_dashboard, "DASHBOARD_COMPONENT_VOLUME_MOUNT", str(overlay))

    run = _new_run("run-1")
    run.metadata["dashboard_components"] = {
        "steps": {**_manifest("steps", "summary_panel", digest), "path": relative}
    }
    run.save()

    with _client(monkeypatch, tmp_path) as client:
        [manifest] = client.get(
            "/api/runs/run-1/dashboard-components", params={"type": "summary_panel"}
        ).json()
        frame = client.get(manifest["frame_url"])

    assert frame.status_code == 200
    csp = frame.headers["content-security-policy"]
    assert "sandbox allow-scripts" in csp
    assert "connect-src 'none'" in csp
    assert "mountViewer" in frame.text


@pytest.mark.skipif(
    shutil.which("node") is None or not (FRONTEND / "node_modules" / "svelte").is_dir(),
    reason="needs node and dashboards/frontend/node_modules (npm ci)",
)
def test_compiled_size_limit(monkeypatch) -> None:
    source = b"<script>let { run } = $props();</script><p>{run?.training_run_id}</p>"
    bundle = _dashboard.compile_dashboard_component_source(source)
    # A minimal component (Svelte runtime included) sits well inside the budget.
    assert len(bundle) < _dashboard.MAX_COMPILED_COMPONENT_BYTES

    monkeypatch.setattr(_dashboard, "MAX_COMPILED_COMPONENT_BYTES", len(bundle) - 1)
    with pytest.raises(
        _dashboard.DashboardComponentCompileError, match="MAX_COMPILED_COMPONENT_BYTES"
    ):
        _dashboard.compile_dashboard_component_source(source)
