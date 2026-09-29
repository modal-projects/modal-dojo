"""Per-tag numeric aggregation across a rollout's samples.

Custom reward-function tags land in each ``Sample.metadata`` dict (see
``sample_extraction.py``'s generic passthrough). ``TrainingRolloutResult``
reduces the numeric ones to count/mean/min/max/p50/p90/p99 at summary time so the
dashboard can chart them over rollouts without a fixed schema.
"""

import pytest

from modal_dojo.common.sample import Sample
from modal_dojo.common.training_rollout import TrainingRolloutResult


def _single(v):
    return {"count": 1, "mean": v, "min": v, "max": v, "p50": v, "p90": v, "p99": v}


def _rollout(samples_metadata):
    return TrainingRolloutResult(
        training_run_id="t",
        rollout_id=0,
        samples=[Sample(metadata=m) for m in samples_metadata],
    )


def test_tag_stats_aggregates_numeric_tags_across_samples():
    rollout = _rollout(
        [
            {"step_K": 2, "task_reward": 1.0},
            {"step_K": 4, "task_reward": 0.0},
            {"step_K": 6, "task_reward": 0.5},
        ]
    )
    stats = rollout.tag_stats
    assert stats["step_K"] == {
        "count": 3,
        "mean": 4.0,
        "min": 2.0,
        "max": 6.0,
        "p50": 4.0,
        "p90": pytest.approx(5.6),
        "p99": pytest.approx(5.96),
    }
    assert stats["task_reward"] == {
        "count": 3,
        "mean": 0.5,
        "min": 0.0,
        "max": 1.0,
        "p50": 0.5,
        "p90": pytest.approx(0.9),
        "p99": pytest.approx(0.99),
    }


def test_tag_stats_excludes_internal_bookkeeping_keys():
    rollout = _rollout([{"response_length": 12, "prompt_length": 4, "custom": 7}])
    stats = rollout.tag_stats
    assert "response_length" not in stats
    assert "prompt_length" not in stats
    assert stats["custom"] == _single(7.0)


def test_tag_stats_ignores_non_numeric_and_bool_values():
    rollout = _rollout([{"task_passed": True, "label": "cat", "score_like": 1}])
    stats = rollout.tag_stats
    assert "task_passed" not in stats
    assert "label" not in stats
    assert stats["score_like"] == _single(1.0)


def test_tag_stats_empty_when_no_numeric_tags():
    assert _rollout([{"label": "cat"}]).tag_stats == {}
    assert _rollout([]).tag_stats == {}


def test_to_summary_includes_tag_stats_only_when_present():
    with_tags = _rollout([{"step_K": 1}])
    assert with_tags.to_summary()["tag_stats"] == {"step_K": _single(1.0)}

    without_tags = _rollout([{"label": "cat"}])
    assert "tag_stats" not in without_tags.to_summary()
