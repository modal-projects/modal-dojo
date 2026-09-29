from __future__ import annotations

from modal_dojo.common import dashboard_components as components_mod
from modal_dojo.common import run as run_mod
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun


def _manifest(name: str, digest: str) -> dict[str, str]:
    return {
        "name": name,
        "type": "trajectory_viewer",
        "path": f"components/trajectory_viewer/{digest}/{name}.svelte",
        "sha256": digest,
    }


def _attach(monkeypatch, run: TrainingRun, name: str, digest: str) -> None:
    manifest = _manifest(name, digest)
    monkeypatch.setattr(
        components_mod, "store_dashboard_component", lambda **_: manifest
    )
    run.add_dashboard_component(
        name=name, component_type="trajectory_viewer", from_path=name, replace=True
    )


def test_component_names_from_independent_handles_are_preserved(
    monkeypatch, fake_volume
) -> None:
    TrainingRun(training_run_id="r", framework=Framework.SLIME, config={}).save()
    first = TrainingRun.from_id("r")
    second = TrainingRun.from_id("r")

    _attach(monkeypatch, first, "a", "a" * 64)
    _attach(monkeypatch, second, "b", "b" * 64)

    stored = TrainingRun.from_id("r").metadata["dashboard_components"]
    assert list(stored) == ["a", "b"]


def test_stale_handle_save_does_not_revert_replacement(
    monkeypatch, fake_volume
) -> None:
    TrainingRun(training_run_id="r", framework=Framework.SLIME, config={}).save()
    original = TrainingRun.from_id("r")
    _attach(monkeypatch, original, "viewer", "a" * 64)

    replacer = TrainingRun.from_id("r")
    _attach(monkeypatch, replacer, "viewer", "b" * 64)

    original.status = run_mod.TrainingRunStatus.COMPLETED
    original.save()

    stored = TrainingRun.from_id("r").metadata["dashboard_components"]
    assert stored["viewer"]["sha256"] == "b" * 64
    assert TrainingRun.from_id("r").status == run_mod.TrainingRunStatus.COMPLETED


def test_replacement_moves_entry_to_end(monkeypatch, fake_volume) -> None:
    TrainingRun(training_run_id="r", framework=Framework.SLIME, config={}).save()
    run = TrainingRun.from_id("r")
    _attach(monkeypatch, run, "a", "a" * 64)
    _attach(monkeypatch, run, "b", "b" * 64)
    _attach(monkeypatch, run, "a", "c" * 64)

    stored = TrainingRun.from_id("r").metadata["dashboard_components"]
    assert list(stored) == ["b", "a"]
    assert stored["a"]["sha256"] == "c" * 64
