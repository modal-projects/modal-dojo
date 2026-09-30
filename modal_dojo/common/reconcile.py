"""Orchestrate orphan cleanup for training runs."""

from __future__ import annotations

from dataclasses import dataclass

from modal_dojo.common.run_reconciler import (
    ReconcileResult,
    reconcile_orphan_runs,
)


@dataclass(frozen=True)
class ReconcileOutcome:
    runs: list[ReconcileResult]
    run_error: str | None = None


def reconcile(*, dry_run: bool = False) -> ReconcileOutcome:
    runs: list[ReconcileResult] = []
    run_error: str | None = None

    try:
        runs = reconcile_orphan_runs(dry_run=dry_run)
    except Exception as exc:
        run_error = str(exc)
        print(f"WARNING: run reconciliation failed: {exc}")

    return ReconcileOutcome(
        runs=runs,
        run_error=run_error,
    )
