"""Tests for the orphan reconciler."""

from __future__ import annotations

from modal_dojo.common.reconcile import reconcile
from modal_dojo.common.run_reconciler import ReconcileResult


def test_reconcile_runs_cleanup(monkeypatch):
    run_result = ReconcileResult(
        training_run_id="run-1",
        reason="stale_modal_app_terminated",
        previous_status="running",
    )
    calls: list[str] = []

    def fake_runs(*, dry_run: bool = False):
        calls.append(f"runs:{dry_run}")
        return [run_result]

    monkeypatch.setattr(
        "modal_dojo.common.reconcile.reconcile_orphan_runs",
        fake_runs,
    )

    outcome = reconcile(dry_run=True)

    assert calls == ["runs:True"]
    assert outcome.runs == [run_result]
    assert outcome.run_error is None


def test_reconcile_isolates_run_failures(monkeypatch):
    def fake_runs(*, dry_run: bool = False):
        raise RuntimeError("runs boom")

    monkeypatch.setattr(
        "modal_dojo.common.reconcile.reconcile_orphan_runs",
        fake_runs,
    )

    outcome = reconcile()

    assert outcome.runs == []
    assert outcome.run_error == "runs boom"
