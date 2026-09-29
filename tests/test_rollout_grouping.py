from types import SimpleNamespace
from typing import Any

import pytest

from modal_dojo.common.sample import Sample
from modal_dojo.common.training_rollout import (
    TrainingRolloutResult,
    TrainingRolloutSample,
    TrainingRolloutSummary,
)
from modal_dojo.frameworks.slime import phase_reporting as pr


EPISODES = [(0.0, 1), (1.0, 2), (2.0, 3), (3.0, 4)]
N_SAMPLES_PER_PROMPT = 2


def _episode_samples():
    samples = []
    for index, (reward, turns) in enumerate(EPISODES):
        for turn in range(turns):
            samples.append(
                SimpleNamespace(
                    index=index,
                    rollout_id=index,
                    prompt=f"episode {index} turn {turn}",
                    response="move",
                    reward=reward,
                )
            )
    return samples


def _capture(monkeypatch, samples, args=None):
    captured = {}
    monkeypatch.setattr(
        pr, "_enqueue_rollout", lambda payload: captured.update(payload)
    )
    pr.report_rollout_samples(
        0,
        args or SimpleNamespace(n_samples_per_prompt=N_SAMPLES_PER_PROMPT),
        samples,
        {},
        None,
    )
    return captured


def _rollout(samples):
    return TrainingRolloutResult(training_run_id="t", rollout_id=0, samples=samples)


def test_payload_tags_each_sample_with_its_rollout(monkeypatch):
    captured = _capture(monkeypatch, _episode_samples())

    # fmt: off
    assert [s["rollout_index"] for s in captured["samples"]] == [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert [s["sample_index"] for s in captured["samples"]] == [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert [s["group_index"] for s in captured["samples"]] == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # fmt: on
    assert captured["n_samples_per_prompt"] == N_SAMPLES_PER_PROMPT


def test_payload_identifies_a_rollout_by_sample_index_when_rollout_id_unset(
    monkeypatch,
):
    samples = [
        SimpleNamespace(index=i, rollout_id=None, prompt="p", response="r", reward=1.0)
        for i in range(3)
    ]
    captured = _capture(monkeypatch, samples)

    assert [s["rollout_index"] for s in captured["samples"]] == [0, 1, 2]


def test_payload_omits_grouping_when_slime_reports_no_indices(monkeypatch):
    samples = [SimpleNamespace(prompt="p", response="r", reward=1.0)]
    captured = _capture(monkeypatch, samples)

    assert "rollout_index" not in captured["samples"][0]
    assert "group_index" not in captured["samples"][0]


def test_grouping_survives_the_rollout_sample_model():
    sample = TrainingRolloutSample.model_validate(
        {"score": 1.0, "rollout_index": 7, "sample_index": 7, "group_index": 3}
    )
    assert sample.rollout_index == 7
    dumped = sample.model_dump(mode="json")
    assert (dumped["sample_index"], dumped["group_index"]) == (7, 3)


def test_samples_per_prompt_reads_as_unknown_when_the_recorder_did_not_report_it():
    reported = TrainingRolloutResult.model_validate(
        {"training_run_id": "t", "rollout_id": 0, "n_samples_per_prompt": 8}
    )
    historical = TrainingRolloutResult.model_validate(
        {"training_run_id": "t", "rollout_id": 0}
    )

    assert reported.n_samples_per_prompt == 8
    assert historical.n_samples_per_prompt is None


def test_grouping_stays_off_the_shared_eval_sample():
    assert not hasattr(Sample(score=1.0), "rollout_index")


def test_rollout_accepts_plain_samples():
    rollout = _rollout([Sample(score=1.0), Sample(score=2.0)])

    assert rollout.mean == 1.5
    assert all(s.rollout_index is None for s in rollout.samples)


def test_mean_averages_rollouts_not_samples():
    samples = [
        TrainingRolloutSample(score=reward, rollout_index=index)
        for index, (reward, turns) in enumerate(EPISODES)
        for _ in range(turns)
    ]
    rollout = _rollout(samples)

    assert rollout.mean == 1.5
    assert rollout.total == 10
    assert rollout.episode_count == 4
    assert rollout.to_summary()["episode_count"] == 4
    assert rollout.to_summary()["total"] == 10


def test_episode_count_reaches_the_dashboard_through_the_summary_model():
    samples = [
        TrainingRolloutSample(score=reward, rollout_index=index)
        for index, (reward, turns) in enumerate(EPISODES)
        for _ in range(turns)
    ]
    summary = TrainingRolloutSummary.model_validate(_rollout(samples).to_summary())

    assert (summary.episode_count, summary.total) == (4, 10)


def test_episode_count_reads_as_unknown_for_records_written_before_it_existed():
    summary = TrainingRolloutSummary.model_validate(
        {
            "training_run_id": "t",
            "rollout_id": 0,
            "created_at": 0,
            "total": 3,
            "mean": 1.0,
        }
    )

    assert summary.episode_count is None


def test_reward_stats_weight_each_episode_once():
    samples = [
        TrainingRolloutSample(score=reward, rollout_index=index)
        for index, (reward, turns) in enumerate(EPISODES)
        for _ in range(turns)
    ]
    rollout = _rollout(samples)

    assert rollout.reward_stats == {
        "count": 4,
        "mean": 1.5,
        "min": 0.0,
        "max": 3.0,
        "p50": 1.5,
        "p90": pytest.approx(2.7),
        "p99": pytest.approx(2.97),
    }
    summary = TrainingRolloutSummary.model_validate(rollout.to_summary())
    assert summary.reward_stats == rollout.reward_stats


def test_reward_stats_absent_for_empty_rollouts_and_old_summaries():
    assert _rollout([]).reward_stats is None
    assert "reward_stats" not in _rollout([]).to_summary()
    summary = TrainingRolloutSummary.model_validate(
        {
            "training_run_id": "t",
            "rollout_id": 0,
            "created_at": 0,
            "total": 0,
            "mean": 0,
        }
    )
    assert summary.reward_stats is None


def test_mean_unchanged_for_one_sample_per_rollout():
    samples = [TrainingRolloutSample(score=float(i), rollout_index=i) for i in range(4)]

    assert _rollout(samples).mean == 1.5


def test_mean_treats_ungrouped_samples_as_their_own_rollouts():
    samples = [
        TrainingRolloutSample(score=score) for score in (0.0, 1.0, 2.0, 3.0, 3.0)
    ]

    assert _rollout(samples).mean == 1.8


def test_mean_falls_back_to_the_rollout_id_in_metadata():
    samples = [
        TrainingRolloutSample(score=reward, metadata={"rollout_id": index})
        for index, (reward, turns) in enumerate(EPISODES)
        for _ in range(turns)
    ]
    rollout = _rollout(samples)

    assert rollout.mean == 1.5
    assert rollout.total == 10
    assert rollout.episode_count == 4


def test_metadata_fallback_treats_rollout_zero_as_a_rollout():
    samples = [
        TrainingRolloutSample(score=score, metadata={"rollout_id": 0})
        for score in (0.0, 4.0)
    ]

    assert _rollout(samples).mean == 2.0


def test_top_level_rollout_index_wins_over_metadata():
    samples = [
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={"rollout_id": 9}),
        TrainingRolloutSample(score=4.0, rollout_index=1, metadata={"rollout_id": 9}),
    ]

    assert _rollout(samples).mean == 2.0


def test_unparseable_metadata_rollout_id_does_not_merge_rollouts():
    samples = [
        TrainingRolloutSample(score=score, metadata={"rollout_id": "episode-a"})
        for score in (0.0, 4.0)
    ]

    assert _rollout(samples).mean == 2.0
    samples.append(TrainingRolloutSample(score=8.0, metadata={"rollout_id": None}))
    assert _rollout(samples).mean == 4.0


def test_tag_stats_weight_each_rollout_once():
    samples = [
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={"step_K": 1.0}),
        TrainingRolloutSample(score=0.0, rollout_index=1, metadata={"step_K": 5.0}),
        TrainingRolloutSample(score=0.0, rollout_index=1, metadata={"step_K": 5.0}),
        TrainingRolloutSample(score=0.0, rollout_index=1, metadata={"step_K": 5.0}),
    ]

    assert _rollout(samples).tag_stats["step_K"] == {
        "count": 2,
        "mean": 3.0,
        "min": 1.0,
        "max": 5.0,
        "p50": 3.0,
        "p90": pytest.approx(4.6),
        "p99": pytest.approx(4.96),
    }


def test_tag_stats_average_a_tag_within_its_rollout():
    samples = [
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={"turn_cost": 0.0}),
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={"turn_cost": 2.0}),
        TrainingRolloutSample(score=0.0, rollout_index=1, metadata={"turn_cost": 4.0}),
    ]

    assert _rollout(samples).tag_stats["turn_cost"] == {
        "count": 2,
        "mean": 2.5,
        "min": 1.0,
        "max": 4.0,
        "p50": 2.5,
        "p90": pytest.approx(3.7),
        "p99": pytest.approx(3.97),
    }


def test_tag_stats_count_only_the_rollouts_that_reported_the_tag():
    samples = [
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={"retries": 2.0}),
        TrainingRolloutSample(score=0.0, rollout_index=0, metadata={}),
        TrainingRolloutSample(score=0.0, rollout_index=1, metadata={}),
    ]

    assert _rollout(samples).tag_stats["retries"] == {
        "count": 1,
        "mean": 2.0,
        "min": 2.0,
        "max": 2.0,
        "p50": 2.0,
        "p90": pytest.approx(2.0),
        "p99": pytest.approx(2.0),
    }


def test_tag_stats_group_historical_records_from_metadata():
    samples = [
        TrainingRolloutSample(metadata={"rollout_id": 0, "step_K": 1.0}),
        TrainingRolloutSample(metadata={"rollout_id": 1, "step_K": 5.0}),
        TrainingRolloutSample(metadata={"rollout_id": 1, "step_K": 5.0}),
    ]

    assert _rollout(samples).tag_stats["step_K"]["mean"] == 3.0


def _regrouped_mean(record: dict) -> float:
    groups: dict[Any, list[float]] = {}
    for position, sample in enumerate(record["samples"]):
        index = sample.get("rollout_index")
        if index is None:
            index = sample.get("metadata", {}).get("rollout_id")
        key = ("rollout", index) if index is not None else ("sample", position)
        groups.setdefault(key, []).append(float(sample["score"]))
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def test_export_keys_reproduce_the_mean_they_ship_with():
    samples = [
        TrainingRolloutSample(score=0.0, rollout_index=0),
        TrainingRolloutSample(score=1.0, rollout_index=1),
        TrainingRolloutSample(score=1.0, rollout_index=1),
        TrainingRolloutSample(score=5.0, metadata={"rollout_id": 2}),
        TrainingRolloutSample(score=5.0, metadata={"rollout_id": 2}),
        TrainingRolloutSample(score=6.0),
    ]
    rollout = _rollout(samples)

    assert rollout.mean == 3.0
    assert _regrouped_mean(rollout.model_dump(mode="json")) == rollout.mean
    assert rollout.to_summary()["mean"] == rollout.mean
