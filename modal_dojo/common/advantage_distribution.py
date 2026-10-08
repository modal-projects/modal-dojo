"""Per-step, per-group advantage distributions for RL training runs.

slime logs only the *mean* advantage per training step. For GRPO-style
estimators the interesting signal is the *spread* of advantages within each
prompt group (the ``n_samples_per_prompt`` responses to one prompt), since the
advantage is a group-normalized reward. This module stores the full per-sample
distribution so the dashboard can render per-group histograms / quantiles.

Write path: slime's ``log_rollout_data`` hook → the async phase-reporter →
``POST /api/advantage-distributions``. Each data-parallel rank holds a disjoint
shard of the rollout's samples, so each rank posts its own shard and we persist
one file per ``(run, rollout, dp_rank)`` — concurrent DP posts never touch the
same file. Reads merge the shards and group by ``group_index``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, Field

from modal_dojo.utils.metadata import (
    MetadataStore,
    vol_list_prefix,
    vol_put,
)

# Quantiles reported per group (and overall). Kept small so payloads stay light.
_QUANTILES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted, non-empty list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def distribution_stats(values: list[float]) -> dict[str, Any]:
    """Summarize a list of advantages: count/mean/std/min/max + quantiles.

    Pure function (no torch / no IO) so it is unit-testable in isolation.
    """
    n = len(values)
    if n == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "quantiles": {f"p{int(q * 100)}": 0.0 for q in _QUANTILES},
        }
    mean = sum(values) / n
    # Population std (matches how a "spread of this step's advantages" reads).
    var = sum((v - mean) ** 2 for v in values) / n
    ordered = sorted(values)
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": ordered[0],
        "max": ordered[-1],
        "quantiles": {f"p{int(q * 100)}": quantile(ordered, q) for q in _QUANTILES},
    }


class AdvantageSample(BaseModel):
    """One sample's group-normalized advantage within a rollout step."""

    # Global sample index within the rollout step (``sample.index`` in slime).
    sample_index: int
    # ``sample_index // n_samples_per_prompt`` — the GRPO prompt group.
    group_index: int
    # Mask-weighted mean advantage over the sample's response tokens. For GRPO
    # the advantage is constant across response tokens, so this is just that
    # constant; for token-varying estimators it is the per-token mean.
    advantage: float
    # The pre-normalization reward, when slime exposes it (handy for sanity).
    raw_reward: float | None = None


class AdvantageDistribution(BaseModel):
    """One data-parallel rank's shard of a step's advantage distribution."""

    training_run_id: str
    rollout_id: int
    # Which DP rank produced this shard; part of the storage key so shards from
    # different ranks for the same step coexist instead of overwriting.
    dp_rank: int = 0
    n_samples_per_prompt: int = 1
    created_at: int = 0
    samples: list[AdvantageSample] = Field(default_factory=list)

    @property
    def storage_key(self) -> str:
        # Zero-padded so a prefix scan returns steps in order, and so dp shards
        # of one step sort together.
        return f"{self.training_run_id}__{self.rollout_id:08d}__dp{self.dp_rank:03d}"

    @staticmethod
    def shard_prefix(training_run_id: str, rollout_id: int) -> str:
        return f"{training_run_id}__{int(rollout_id):08d}__dp"

    @staticmethod
    def run_prefix(training_run_id: str) -> str:
        return f"{training_run_id}__"

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        return vol_put(
            MetadataStore.ADVANTAGE_DISTRIBUTIONS,
            self.storage_key,
            self.model_dump(mode="json"),
            is_async=is_async,
        )

    @classmethod
    def merged_for_step(
        cls, training_run_id: str, rollout_id: int
    ) -> dict[str, Any] | None:
        """Merge every DP shard of one step into per-group distributions.

        Returns ``None`` when no shard exists for the step yet.
        """
        shards = vol_list_prefix(
            MetadataStore.ADVANTAGE_DISTRIBUTIONS,
            cls.shard_prefix(training_run_id, rollout_id),
        )
        if not shards:
            return None
        return merge_shards(shards)

    @classmethod
    def list_steps_for_run(cls, training_run_id: str) -> list[dict[str, Any]]:
        """One lightweight summary row per step that has advantage data."""
        shards = vol_list_prefix(
            MetadataStore.ADVANTAGE_DISTRIBUTIONS, cls.run_prefix(training_run_id)
        )
        return summarize_steps(shards)


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-DP-rank shards of one step into grouped distributions.

    Pure function over plain dicts so it is unit-testable without IO. Samples
    are de-duplicated by ``sample_index`` (a rank should never emit a duplicate,
    but a retried POST might), then grouped by ``group_index``.
    """
    training_run_id = ""
    rollout_id = 0
    n_samples_per_prompt = 1
    created_at = 0
    by_index: dict[int, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        training_run_id = (
            shard.get("training_run_id", training_run_id) or training_run_id
        )
        rollout_id = int(shard.get("rollout_id", rollout_id) or rollout_id)
        n_samples_per_prompt = int(
            shard.get("n_samples_per_prompt", n_samples_per_prompt)
            or n_samples_per_prompt
        )
        created_at = max(created_at, int(shard.get("created_at", 0) or 0))
        for sample in shard.get("samples") or []:
            if not isinstance(sample, dict) or sample.get("sample_index") is None:
                continue
            by_index[int(sample["sample_index"])] = sample

    groups: dict[int, dict[str, list[Any]]] = {}
    for sample in by_index.values():
        gi = int(sample.get("group_index", 0) or 0)
        bucket = groups.setdefault(gi, {"advantages": [], "raw_rewards": []})
        bucket["advantages"].append(float(sample.get("advantage", 0.0) or 0.0))
        rr = sample.get("raw_reward")
        if rr is not None:
            bucket["raw_rewards"].append(float(rr))

    group_rows = []
    for gi in sorted(groups):
        advs = groups[gi]["advantages"]
        raw = groups[gi]["raw_rewards"]
        row: dict[str, Any] = {
            "group_index": gi,
            "advantages": advs,
            "stats": distribution_stats(advs),
        }
        if raw:
            row["raw_rewards"] = raw
        group_rows.append(row)

    all_advs = [a for g in group_rows for a in g["advantages"]]
    return {
        "training_run_id": training_run_id,
        "rollout_id": rollout_id,
        "n_samples_per_prompt": n_samples_per_prompt,
        "created_at": created_at,
        "num_groups": len(group_rows),
        "num_samples": len(all_advs),
        "overall": distribution_stats(all_advs),
        "groups": group_rows,
    }


def summarize_steps(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per step (rollout_id), each carrying the step's overall
    advantage distribution stats.

    Including ``stats`` (mean/std/min/max + quantiles) here lets the dashboard
    draw the "distribution over time" fan chart from a single list request,
    rather than fetching every step's full per-group payload.
    """
    by_step: dict[int, list[dict[str, Any]]] = {}
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        by_step.setdefault(int(shard.get("rollout_id", 0) or 0), []).append(shard)

    rows: list[dict[str, Any]] = []
    for rollout_id in sorted(by_step):
        merged = merge_shards(by_step[rollout_id])
        rows.append(
            {
                "training_run_id": merged["training_run_id"],
                "rollout_id": rollout_id,
                "created_at": merged["created_at"],
                "num_samples": merged["num_samples"],
                "num_groups": merged["num_groups"],
                "n_samples_per_prompt": merged["n_samples_per_prompt"],
                "stats": merged["overall"],
            }
        )
    return rows
