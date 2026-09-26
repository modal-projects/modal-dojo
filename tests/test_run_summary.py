from __future__ import annotations

import pytest

from modal_training_gym.common import run_summary as run_summary_module
from modal_training_gym.common.run_summary import (
    build_run_summaries,
    build_run_summary,
)


def _run(**overrides):
    run = {
        "training_run_id": "run-1",
        "framework": "slime",
        "status": "running",
        "modal_app_id": "ap-123",
        "created_at": 100,
        "started_at": 110,
        "updated_at": 150,
        "config": {
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "dataset": {"hf_repo": "openai/gsm8k"},
            "recipe": {
                "gpu_type": "H100",
                "actor_num_nodes": 2,
                "actor_num_gpus_per_node": 8,
            },
            "wandb": {
                "entity": "modal",
                "project": "training",
                "group": "smoke",
            },
            "lr": 1e-6,
            "global_batch_size": 32,
        },
        "metadata": {
            "group_id": "sweep-a",
            "framework_progress": {
                "phase": "generate_rollouts",
                "current": 4,
                "total": 10,
                "unit": "step",
                "is_active": True,
                "rollout_id": 3,
                "updated_at": 149,
            },
            "latest_rollout": {
                "rollout_id": 3,
                "mean": 0.75,
                "total": 16,
                "created_at": 148,
            },
            "group_tags": {
                "group_id": "sweep-a",
                "axes": ["recipe.lr"],
                "overrides": {"recipe.lr": 1e-6},
                "tags": [],
            },
            "attempt_count": 2,
            "last_attempt_status": "running",
            "last_attempt_started_at": 140,
            "resumed_from_checkpoint": True,
            "resume_checkpoint_name": "iter_0000003",
            "resume_from_iteration": 3,
            "wandb_attempts": [
                {
                    "attempt": 2,
                    "entity": "modal",
                    "project": "training",
                    "run_id": "run-1-a2",
                }
            ],
        },
    }
    run.update(overrides)
    return run


def _result(**overrides):
    result = {
        "training_run_id": "run-1",
        "app_name": "training-app",
        "checkpoint_dir": "/checkpoints/run-1",
        "model_config": {
            "model_name": "Qwen/Qwen3-4B",
            "model_path": "/checkpoints/run-1/iter_0000004",
        },
        "wandb_entity": "modal",
        "wandb_project": "training",
        "wandb_training_run_id": "run-1-a2",
    }
    result.update(overrides)
    return result


def test_legacy_value_helpers_unwrap_and_normalize_values():
    wrapped = {"value": "legacy", "source": "old-store"}

    assert run_summary_module._unwrap(wrapped) == "legacy"
    assert run_summary_module._text(wrapped) == "legacy"
    assert run_summary_module._text(None) == ""
    assert run_summary_module._mapping({"value": {"key": "value"}}) == {"key": "value"}
    assert run_summary_module._mapping("not-a-mapping") == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        (123.5, 123),
        ("123", 123),
        ("123.5", 123),
        ({"value": "123.5", "source": "legacy"}, 123),
        ("2025-01-02T03:04:05Z", 1735787045),
        ("2025-01-02T03:04:05", 1735787045),
        ("", 0),
        ("not-a-timestamp", 0),
        (None, 0),
        (True, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        ("1e309", 0),
    ],
)
def test_timestamp_normalization(value, expected):
    assert run_summary_module._timestamp(value) == expected


def test_number_and_integer_helpers_apply_fallbacks():
    assert run_summary_module._number({"value": "2.5"}) == 2.5
    assert run_summary_module._number("invalid", default=7) == 7
    assert run_summary_module._number(float("inf"), default=7) == 7
    assert run_summary_module._optional_int("4.0") == 4
    assert run_summary_module._optional_int("4.25") is None
    assert run_summary_module._optional_int("") is None
    assert run_summary_module._optional_int("invalid") is None
    assert run_summary_module._integer("invalid", default=7) == 7


def test_url_helpers_match_old_frontend_behavior():
    assert run_summary_module._modal_app_url("") is None
    assert run_summary_module._modal_app_url("ap-123") == "https://modal.com/id/ap-123"
    assert (
        run_summary_module._modal_app_url("https://modal.com/apps/example")
        == "https://modal.com/apps/example"
    )
    assert run_summary_module._wandb_url(None, "project", "run") is None
    assert (
        run_summary_module._wandb_url("my entity", "project/name", "run id")
        == "https://wandb.ai/my%20entity/project%2Fname/runs/run%20id"
    )
    assert (
        run_summary_module._wandb_url("entity", "project", "")
        == "https://wandb.ai/entity/project"
    )


def test_config_summary_fallbacks_and_metric_defaults():
    summary = run_summary_module._config_summary(
        {
            "model": {"model_name": "model"},
            "dataset": {
                "hf_repo": "repo/dataset",
                "prompt_data": "prompts.jsonl",
                "name": "fallback-name",
            },
            "preset": {"gpu_type": "A100"},
            "recipe": {"gpu_type": "H100"},
            "wandb": {
                "entity": "entity",
                "project": "project",
                "group": "group",
            },
        },
        "abcdefghijk",
    )

    assert summary.model_name == "model"
    assert summary.dataset_name == "repo/dataset"
    assert summary.dataset_prompt_data == "prompts.jsonl"
    assert summary.gpu_type == "H100"
    assert summary.loss_type == ""
    assert summary.lr == 0
    assert summary.global_batch_size == 0
    assert summary.metric_run_id == "abcdefgh"
    assert summary.metric_url == "https://wandb.ai/entity/project/runs/abcdefgh"
    assert [link.label for link in summary.metric_links] == ["Metric"]
    payload = summary.model_dump(mode="json")
    assert payload["wandb_training_run_id"] == "abcdefgh"
    assert payload["wandb_url"] == summary.metric_url
    assert [link["label"] for link in payload["wandb_links"]] == ["W&B"]
    assert run_summary_module._config_summary(None, "run-id") == {}
    sft = run_summary_module._config_summary(
        {"recipe": {"loss_type": "sft_loss"}},
        "run-id",
    )
    assert sft.loss_type == "sft_loss"


def test_progress_rollout_and_resume_helpers_handle_missing_and_invalid_values():
    assert run_summary_module._framework_progress({}) is None
    progress = run_summary_module._framework_progress(
        {
            "framework_progress": {
                "current": "invalid",
                "total": 0,
                "unit": "",
                "is_active": "true",
                "rollout_id": "2.5",
                "step_id": "3",
                "updated_at": "2025-01-02T03:04:05Z",
            }
        }
    )
    assert progress.current is None
    assert progress.total is None
    assert progress.unit == "step"
    assert progress.is_active is None
    assert progress.rollout_id is None
    assert progress.step_id == 3
    assert progress.updated_at == 1735787045

    assert run_summary_module._latest_rollout({}) is None
    rollout = run_summary_module._latest_rollout(
        {
            "latest_rollout": {
                "rollout_id": "invalid",
                "mean": "invalid",
                "total": "3.5",
                "created_at": "100.5",
            }
        }
    )
    assert rollout.rollout_id == 0
    assert rollout.mean == 0
    assert rollout.total == 0
    assert rollout.created_at == 100

    assert run_summary_module._resume_state({"attempt_count": 1}) is None
    resume = run_summary_module._resume_state(
        {
            "attempt_count": 1,
            "resume_checkpoint_path": "/checkpoints/iter_1",
            "resume_from_iteration": "1.5",
            "last_attempt_started_at": "100.5",
        }
    )
    assert resume.resumed_from_checkpoint is True
    assert resume.resume_from_iteration is None
    assert resume.last_attempt_started_at == 100


def test_group_tag_helper_synthesizes_tags_from_overrides():
    assert run_summary_module._group_tags({}, "") is None

    tags = run_summary_module._group_tags(
        {
            "group_tags": {
                "overrides": {
                    "recipe.actor_num_nodes": 2,
                    "config.lr": 1e-6,
                }
            }
        },
        "group-1",
    )

    assert tags.group_id == "group-1"
    assert tags.axes == ["recipe.actor_num_nodes", "config.lr"]
    assert [(tag.key, tag.label, tag.value) for tag in tags.tags] == [
        ("recipe.actor_num_nodes", "actor num nodes", 2),
        ("config.lr", "lr", 1e-6),
    ]


def test_metric_attempt_helpers_skip_invalid_links_and_dedupe_by_url():
    links = run_summary_module._metric_attempt_links(
        {
            "wandb_attempts": [
                None,
                {"attempt": 1, "project": "missing-entity"},
                {
                    "attempt": "2",
                    "entity": "entity",
                    "project": "project",
                    "run_id": "run-2",
                },
            ]
        }
    )
    duplicate = run_summary_module.MetricLink(
        label="duplicate",
        url=links[0].url,
    )
    other = run_summary_module.MetricLink(
        label="other",
        url="https://wandb.ai/entity/project/runs/other",
    )

    assert links[0].label == "Metric a2"
    assert links[0].attempt == 2
    assert run_summary_module._dedupe_links(links, [duplicate, other]) == [
        links[0],
        other,
    ]


def test_metric_summary_uses_provider_neutral_config():
    metric = {
        "provider": "wandb",
        "project": "training",
        "entity": "modal",
        "run_id": "run-1",
        "url": "https://wandb.ai/modal/training/runs/run-1",
    }
    run = _run(config={"metrics": metric}, metadata={"metric_attempts": [metric]})

    summary = build_run_summary(run)

    assert summary.config_summary.metric_provider == "wandb"
    assert summary.config_summary.metric_url == metric["url"]
    assert [(link.label, link.url) for link in summary.metric_links] == [
        ("Metric", metric["url"])
    ]


def test_missing_identity_uses_stable_fallbacks():
    summary = build_run_summary(
        {
            "modal_app_id": "ap-fallback",
            "created_at": "invalid",
            "config": None,
        },
        fallback_index=3,
    )

    assert summary.training_run_id == "unknown-run-ap-fallback-0-3"
    assert summary.run_id == summary.training_run_id
    assert summary.status == "running"
    assert summary.framework == "(untagged)"
    assert summary.created_at == 0
    assert summary.started_at == 0
    assert summary.updated_at == 0
    assert summary.config_summary == {}


def test_build_current_summary_joins_result_and_derives_public_fields():
    summary = build_run_summary(
        _run(framework_status="generate_rollouts"),
        _result(),
    )

    assert summary.training_run_id == "run-1"
    assert summary.run_id == "run-1"
    assert summary.status == "running"
    assert summary.display_status == "completed"
    assert summary.display_stage == "Generating rollouts"
    assert summary.model == "Qwen/Qwen3-4B"
    assert summary.dataset == "openai/gsm8k"
    assert summary.recipe == "slime"
    assert summary.group_id == "sweep-a"
    assert summary.framework_progress.current == 4
    assert summary.framework_progress.is_active is True
    assert summary.latest_rollout.mean == 0.75
    assert summary.has_train_result is True
    assert summary.train_result.checkpoint_dir == "/checkpoints/run-1"
    assert summary.modal_app_url == "https://modal.com/id/ap-123"
    assert summary.resume_state.attempt_count == 2
    assert summary.group_tags.tags[0].key == "recipe.lr"
    assert [link.label for link in summary.metric_links] == ["Metric a2", "Metric"]
    payload = summary.model_dump(mode="json")
    assert [link["label"] for link in payload["wandb_links"]] == ["W&B a2", "W&B"]
    assert payload["train_result"]["metric_url"] == payload["train_result"]["wandb_url"]


def test_build_historical_summary_accepts_aliases_wrappers_and_iso_timestamps():
    historical = {
        "run_id": "legacy-1",
        "framework": {"value": "miles"},
        "status": {"value": "completed"},
        "created_at": "2025-01-02T03:04:05Z",
        "ended_at": "1735787105",
        "config": {
            "model": {"model_name": {"value": "legacy/model"}},
            "dataset": {"prompt_data": "legacy.jsonl"},
            "preset": {"gpu_type": "A100"},
        },
    }

    summary = build_run_summary(historical)

    assert summary.training_run_id == "legacy-1"
    assert summary.framework == "miles"
    assert summary.status == "completed"
    assert summary.model == "legacy/model"
    assert summary.dataset == "legacy.jsonl"
    assert summary.config_summary.gpu_type == "A100"
    assert summary.created_at == 1735787045
    assert summary.completed_at == 1735787105
    assert summary.duration_seconds == 60


def test_missing_result_keeps_run_available():
    summary = build_run_summaries([_run()], train_results=None)[0]

    assert summary.status == "running"
    assert summary.display_status == "pending"
    assert summary.has_train_result is False
    assert summary.train_result is None


def test_display_stage_marks_inactive_gpu_stage_as_queued():
    run = _run(
        framework_status="download_model",
        metadata={
            "framework_progress": {
                "is_active": False,
            }
        },
    )

    summary = build_run_summary(run)

    assert summary.display_stage == "Queuing for GPU — Downloading model"


def test_build_summaries_dedupes_by_id_and_prefers_newest_record():
    older = _run(created_at=90, updated_at=90)
    newer = _run(created_at=200, updated_at=210, framework_status="training")

    summaries = build_run_summaries([older, newer], [_result()])

    assert len(summaries) == 1
    assert summaries[0].created_at == 200
    assert summaries[0].framework_status == "training"


def test_missing_config_and_urls_match_old_frontend_defaults():
    summary = build_run_summary(
        _run(config=None, modal_app_id=""),
        _result(
            wandb_entity=None,
            wandb_project=None,
            wandb_training_run_id=None,
        ),
    )

    assert summary.config_summary == {}
    assert summary.modal_app_url is None
    assert summary.train_result.metric_url is None
    payload = summary.model_dump(mode="json")
    assert payload["train_result"]["wandb_url"] is None


def test_config_defaults_and_fractional_integer_fields_are_rejected():
    run = _run(
        config={},
        metadata={
            "framework_progress": {"current": "1.5", "total": "2.5"},
            "latest_rollout": {"rollout_id": "1.5", "total": "3.5"},
            "attempt_count": "2.5",
            "resume_from_iteration": "1.5",
        },
    )

    summary = build_run_summary(run)

    assert summary.config_summary.lr == 0
    assert summary.config_summary.metric_url is None
    payload = summary.model_dump(mode="json")
    assert payload["config_summary"]["wandb_url"] is None
    assert summary.framework_progress.current is None
    assert summary.framework_progress.total is None
    assert summary.latest_rollout.rollout_id == 0
    assert summary.latest_rollout.total == 0
    assert summary.resume_state is None


def test_negative_duration_is_recalculated():
    summary = build_run_summary(
        _run(started_at=110, ended_at=125, duration_seconds=-10)
    )

    assert summary.duration_seconds == 15


def test_wrappers_with_additional_keys_are_unwrapped():
    summary = build_run_summary(
        _run(
            framework={"value": "miles", "source": "legacy"},
            created_at="100.5",
            modal_app_id="https://modal.com/apps/example",
        )
    )

    assert summary.framework == "miles"
    assert summary.created_at == 100
    assert summary.modal_app_url == "https://modal.com/apps/example"


def test_malformed_records_do_not_hide_valid_runs():
    summaries = build_run_summaries(
        [
            _run(training_run_id="valid-run"),
            _run(
                training_run_id="invalid-run",
                step_times={"1": {"phase": {"not": "an integer"}}},
            ),
            None,  # type: ignore[list-item]
        ],
        [None],  # type: ignore[list-item]
    )

    assert [summary.training_run_id for summary in summaries] == ["valid-run"]
