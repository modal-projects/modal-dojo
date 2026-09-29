"""training-gym cleanup — delete old failed run metadata from the shared volume."""

from __future__ import annotations

import time

from modal.exception import InvalidError, NotFoundError

from modal_training_gym.common.advantage_distribution import AdvantageDistribution
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.step_timing import RoleTimingRecord
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put_summary_items,
    vol_remove,
    vol_remove_keys_with_prefix,
)


TERMINAL_STATUSES = frozenset({TrainingRunStatus.FAILED, TrainingRunStatus.CANCELLED})


def cleanup(*, older_than_days: int = 7, dry_run: bool = False) -> None:
    cutoff = int(time.time()) - older_than_days * 86400

    print(f"Finding failed/cancelled runs older than {older_than_days} days …")

    raw_runs = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    runs: list[TrainingRun] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            continue
        if "training_run_id" not in raw and "run_id" in raw:
            raw["training_run_id"] = raw["run_id"]
        try:
            runs.append(TrainingRun.from_stored_data(raw))
        except Exception:
            continue

    targets = [
        r
        for r in runs
        if r.status in TERMINAL_STATUSES
        and (r.created_at or r.started_at or 0) < cutoff
        and (r.created_at or r.started_at or 0) > 0
    ]

    if not targets:
        print("Nothing to clean up.")
        return

    target_ids = {r.training_run_id for r in targets}
    print(
        f"{'Would delete' if dry_run else 'Deleting'} {len(targets)} terminal run(s):\n"
    )
    for r in sorted(targets, key=lambda r: r.created_at or 0):
        age_days = (time.time() - (r.created_at or r.started_at)) / 86400
        print(f"  {r.training_run_id}  ({age_days:.0f}d ago, {r.status.value})")

    if dry_run:
        print("\nDry run — no changes made. Remove --dry-run to delete.")
        return

    deleted_runs = 0
    deleted_rollouts = 0
    deleted_tokens = 0

    for r in targets:
        rid = r.training_run_id
        if vol_remove(MetadataStore.TRAINING_RUNS, rid):
            deleted_runs += 1
        vol_remove(MetadataStore.FRAMEWORK_STATUS_TOKENS, rid)
        vol_remove_keys_with_prefix(MetadataStore.TRAINING_RUN_UPDATES, f"{rid}__")
        vol_remove_keys_with_prefix(
            MetadataStore.ADVANTAGE_DISTRIBUTIONS,
            AdvantageDistribution.run_prefix(rid),
        )
        timing_volume = metadata._metadata_volume()
        metadata._safe_reload(timing_volume)
        try:
            timing_volume.remove_file(RoleTimingRecord.store(rid), recursive=True)
        except (FileNotFoundError, NotFoundError):
            pass
        except InvalidError as exc:
            if "No such file or directory" not in str(exc):
                raise
        deleted_tokens += 1

    rollout_summary = (
        vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
    )
    kept_rollout_items = []
    for item in rollout_summary:
        if item.get("training_run_id") in target_ids:
            key = item.get("summary_key") or ""
            if key:
                vol_remove(MetadataStore.TRAINING_ROLLOUTS, key)
                deleted_rollouts += 1
        else:
            kept_rollout_items.append(item)
    if deleted_rollouts:
        vol_put_summary_items(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY, kept_rollout_items
        )

    run_summary = vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) or []
    kept_run_items = [
        item for item in run_summary if item.get("training_run_id") not in target_ids
    ]
    if len(kept_run_items) != len(run_summary):
        vol_put_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, kept_run_items)

    print(
        f"\nDone: {deleted_runs} run(s), {deleted_rollouts} rollout(s), "
        f"{deleted_tokens} token(s) removed."
    )
