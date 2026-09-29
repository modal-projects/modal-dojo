"""Per-step, per-group advantage distribution capture.

Covers the pure, torch-free seams of the Approach-2 advantage tracker:

  * :func:`_advantage_samples_payload` — turning masked ``(sum, count)`` pairs
    into per-sample rows with the right GRPO ``group_index`` and divide-by-count
    (incl. the empty-mask guard);
  * :func:`distribution_stats` — count/mean/std/min/max + quantiles; and
  * :func:`merge_shards` — combining per-DP-rank shards into per-group
    distributions, de-duplicating by ``sample_index``.

The build-time patch's anchor string is also asserted against the reporter so a
rename can't silently break injection.
"""

import math

from modal_dojo.common.advantage_distribution import (
    distribution_stats,
    merge_shards,
    summarize_steps,
)
from modal_dojo.frameworks.slime import phase_reporting as pr


# ── _advantage_samples_payload ───────────────────────────────────────────────


def test_payload_divides_sum_by_count_and_assigns_groups():
    """advantage = sum/count; group_index = sample_index // n_samples_per_prompt."""
    rows = pr._advantage_samples_payload(
        sample_sums=[2.0, 4.0, 9.0, 3.0],
        sample_counts=[2.0, 2.0, 3.0, 1.0],
        sample_indices=[0, 1, 2, 3],
        raw_rewards=[1.0, 1.0, 0.0, 0.0],
        n_samples_per_prompt=2,
    )
    assert [r["advantage"] for r in rows] == [1.0, 2.0, 3.0, 3.0]
    # indices 0,1 -> group 0 ; indices 2,3 -> group 1
    assert [r["group_index"] for r in rows] == [0, 0, 1, 1]
    assert [r["sample_index"] for r in rows] == [0, 1, 2, 3]
    assert [r["raw_reward"] for r in rows] == [1.0, 1.0, 0.0, 0.0]


def test_payload_zero_count_is_zero_not_nan():
    """An all-masked sample (count 0) yields advantage 0.0, never a division error."""
    rows = pr._advantage_samples_payload(
        sample_sums=[0.0],
        sample_counts=[0.0],
        sample_indices=[5],
        raw_rewards=[None],
        n_samples_per_prompt=4,
    )
    assert rows[0]["advantage"] == 0.0
    assert rows[0]["group_index"] == 1  # 5 // 4
    assert rows[0]["raw_reward"] is None


def test_payload_uses_global_sample_index_for_groups():
    """Group is derived from the global sample index, not the local position."""
    rows = pr._advantage_samples_payload(
        sample_sums=[1.0, 1.0],
        sample_counts=[1.0, 1.0],
        sample_indices=[8, 9],  # a shard that holds only group 4's samples
        raw_rewards=[],
        n_samples_per_prompt=2,
    )
    assert [r["group_index"] for r in rows] == [4, 4]


# ── distribution_stats ───────────────────────────────────────────────────────


def test_distribution_stats_basic():
    stats = distribution_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["mean"] == 2.5
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert math.isclose(stats["std"], math.sqrt(1.25))
    assert stats["quantiles"]["p50"] == 2.5
    assert stats["quantiles"]["p0"] == 1.0
    assert stats["quantiles"]["p100"] == 4.0


def test_distribution_stats_empty():
    stats = distribution_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0
    assert stats["quantiles"]["p50"] == 0.0


# ── merge_shards ─────────────────────────────────────────────────────────────


def _shard(dp_rank, samples):
    return {
        "training_run_id": "run-x",
        "rollout_id": 7,
        "dp_rank": dp_rank,
        "n_samples_per_prompt": 2,
        "created_at": 100 + dp_rank,
        "samples": samples,
    }


def test_merge_shards_groups_across_dp_ranks():
    """Two DP shards holding halves of two groups merge into 2 grouped rows."""
    shard0 = _shard(
        0,
        [
            {"sample_index": 0, "group_index": 0, "advantage": 1.0, "raw_reward": 1.0},
            {"sample_index": 1, "group_index": 0, "advantage": -1.0, "raw_reward": 0.0},
        ],
    )
    shard1 = _shard(
        1,
        [
            {"sample_index": 2, "group_index": 1, "advantage": 0.5, "raw_reward": 1.0},
            {"sample_index": 3, "group_index": 1, "advantage": -0.5, "raw_reward": 0.0},
        ],
    )
    merged = merge_shards([shard0, shard1])

    assert merged["training_run_id"] == "run-x"
    assert merged["rollout_id"] == 7
    assert merged["num_groups"] == 2
    assert merged["num_samples"] == 4
    assert merged["created_at"] == 101  # max across shards

    g0, g1 = merged["groups"]
    assert g0["group_index"] == 0
    assert sorted(g0["advantages"]) == [-1.0, 1.0]
    assert g0["stats"]["mean"] == 0.0
    assert g1["group_index"] == 1
    assert sorted(g1["advantages"]) == [-0.5, 0.5]


def test_merge_shards_dedupes_by_sample_index():
    """A retried POST (same sample_index twice) must not double-count."""
    s = {"sample_index": 0, "group_index": 0, "advantage": 2.0}
    merged = merge_shards([_shard(0, [s]), _shard(0, [s])])
    assert merged["num_samples"] == 1
    assert merged["groups"][0]["advantages"] == [2.0]


def test_summarize_steps_one_row_per_rollout():
    shards = [
        _shard(0, [{"sample_index": 0, "group_index": 0, "advantage": 1.0}]),
        _shard(1, [{"sample_index": 1, "group_index": 0, "advantage": 1.0}]),
        {
            **_shard(0, [{"sample_index": 9, "group_index": 4, "advantage": 1.0}]),
            "rollout_id": 8,
        },
    ]
    rows = summarize_steps(shards)
    assert [r["rollout_id"] for r in rows] == [7, 8]
    assert rows[0]["num_samples"] == 2  # both dp shards of step 7
    assert rows[1]["num_samples"] == 1
    # Each row carries the step's overall distribution stats for the fan chart.
    assert rows[0]["stats"]["count"] == 2
    assert rows[0]["stats"]["mean"] == 1.0
    assert "p50" in rows[0]["stats"]["quantiles"]


# ── patch anchor guard ───────────────────────────────────────────────────────


def test_patch_anchor_targets_existing_reporter():
    """The injected call must name a function that actually exists, so a rename
    surfaces here instead of as a silent no-op patch in the container."""
    from modal_dojo.frameworks.slime.modal_helpers.patches import (
        patch_advantage_distribution as patch,
    )

    assert "report_advantage_distribution" in patch.PREAMBLE
    assert hasattr(pr, "report_advantage_distribution")
    # The injection keeps the original anchor intact (so slime's own logic runs).
    assert patch.ANCHOR in patch.INJECTION
    assert "_tg_report_advantage_distribution(rollout_id, args, rollout_data)" in (
        patch.INJECTION
    )
