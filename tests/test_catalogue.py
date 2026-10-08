from __future__ import annotations

import pytest

import modal_dojo
from modal_dojo.common.run_summary import GroupTags, RunSummary
from modal_dojo.common.models.validation import _ValidationConfig
from modal_dojo.common import catalogue
from modal_dojo.common.catalogue import (
    CATALOGUE,
    CatalogueEntry,
    AUTOCONFIG_ENTRY_TAG,
    REWARD_KEY,
    STEPS,
    build_catalogue,
    entry_run_ids,
    context_length,
    gpu_hour_cost,
    logged_per_step,
    recipe_defaults,
    run_fields,
    step_rows,
    step_seconds,
)


def test_step_seconds_uses_measured_steps() -> None:
    measured = {
        "10": {"duration_s": 300.0, "partial": False},
        "2": {"duration_s": 200.0, "partial": False},
        "1": {"duration_s": 900.0, "partial": False},
        "11": {"duration_s": 50.0, "partial": True},
    }
    table = {0: {"perf/step_time": 1.0}}

    assert step_seconds(measured, table) == {10: 300.0, 2: 200.0, 1: 900.0}


def test_step_seconds_falls_back_to_framework_step_time() -> None:
    # Miles images without the timing patch record no substep timing.
    table = {
        2: {"perf/step_time": 2127.3, "rollout/step": 0.0},
        0: {REWARD_KEY: 0.5},
        5: {"perf/step_time": 1800.0},
    }

    assert step_seconds({}, table) == {1: 2127.3, 2: 1800.0}


def test_rewards_are_numbered_by_logging_order() -> None:
    # A mirrored row's rollout/step can belong to another logging call.
    table = {
        0: {REWARD_KEY: 84.9, "rollout/step": 0.0},
        3: {REWARD_KEY: 85.7, "rollout/step": 2.0},
        4: {"train/step": 1.0},
        6: {REWARD_KEY: 91.1, "rollout/step": 5.0},
    }

    assert logged_per_step(table, REWARD_KEY) == {1: 84.9, 2: 85.7, 3: 91.1}


def test_step_rows_price_each_step_and_keep_a_running_total() -> None:
    rows = step_rows(
        {1: 3600.0, 2: 1800.0},
        {1: 0.25, 2: 0.5, 3: 0.75},
        gpus=8,
        gpu_hour_usd=4.0,
    )

    assert rows == [
        {
            "step": 1,
            "seconds": 3600.0,
            "cost_usd": 32.0,
            "total_usd": 32.0,
            "raw_reward": 0.25,
        },
        {
            "step": 2,
            "seconds": 1800.0,
            "cost_usd": 16.0,
            "total_usd": 48.0,
            "raw_reward": 0.5,
        },
        {
            "step": 3,
            "seconds": None,
            "cost_usd": None,
            "total_usd": None,
            "raw_reward": 0.75,
        },
    ]


def test_step_rows_total_sums_unrounded_costs() -> None:
    # Three steps of $0.333... each total $1.00, not the $0.99 of their rounded parts.
    rows = step_rows({1: 1.0, 2: 1.0, 3: 1.0}, {}, gpus=1, gpu_hour_usd=1200.0)

    assert [row["cost_usd"] for row in rows] == [0.33, 0.33, 0.33]
    assert rows[-1]["total_usd"] == 1.0


def test_step_rows_without_a_price_still_report_time_and_reward() -> None:
    rows = step_rows({1: 60.0}, {1: 0.5}, gpus=8, gpu_hour_usd=None)

    assert rows == [
        {
            "step": 1,
            "seconds": 60.0,
            "cost_usd": None,
            "total_usd": None,
            "raw_reward": 0.5,
        }
    ]


def test_context_length_is_the_cap_that_binds_first() -> None:
    assert context_length({"rollout_max_response_len": 16384}) == 16384
    # A context cap bounds prompt plus response, so the smaller cap binds.
    assert (
        context_length(
            {"rollout_max_response_len": 32768, "rollout_max_context_len": 8192}
        )
        == 8192
    )
    assert (
        context_length(
            {"rollout_max_response_len": 16384, "rollout_max_context_len": 32768}
        )
        == 16384
    )
    assert context_length({"extra_config": {"rollout_max_response_len": 2048}}) == 2048
    assert context_length({}) is None


def test_planned_context_length_follows_the_recipe_defaults() -> None:
    assert context_length(recipe_defaults("GLM_5_3_Flash_LoRA_Recipe")) == 32768
    assert context_length(recipe_defaults("Qwen3_6_27B_Recipe")) == 4096


@pytest.mark.parametrize(
    ("gpu_type", "expected"),
    [("H200", 4.54), ("B300:8", 7.1), ("A100-80GB", 2.5), ("TPU", None)],
)
def test_gpu_hour_cost_matches_billing_rate_keys(gpu_type, expected) -> None:
    rates = {
        "gpu_hour_cost_h200": "4.54000",
        "gpu_hour_cost_b300": "7.10000",
        "gpu_hour_cost_a100_80gb": "2.50000",
    }

    assert gpu_hour_cost(rates, gpu_type) == expected


@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda entry: entry.name)
def test_catalogue_entries_name_registered_models_and_public_recipes(entry) -> None:
    _ValidationConfig.find(entry.name)
    assert entry.recipe in modal_dojo.__all__


def _sweep_summary(run_id: str, entry: str, created_at: int) -> RunSummary:
    return RunSummary(
        training_run_id=run_id,
        run_id=run_id,
        created_at=created_at,
        group_id="autoconfig-lr-abc123",
        group_tags=GroupTags(
            group_id="autoconfig-lr-abc123",
            axes=["recipe.lr"],
            overrides={AUTOCONFIG_ENTRY_TAG: entry, "recipe.lr": 1e-6},
        ),
    )


def test_entry_run_ids_take_pinned_runs_then_sweep_runs_newest_first() -> None:
    entry = CATALOGUE[0]
    summaries = [
        _sweep_summary("old", entry.name, 1),
        _sweep_summary("new", entry.name, 2),
        _sweep_summary("other", CATALOGUE[1].name, 3),
        RunSummary(
            training_run_id="manual", run_id="manual", created_at=4, group_id="my-sweep"
        ),
    ]
    pinned = CatalogueEntry(entry.name, entry.recipe, run_ids=("pinned", "new"))

    assert entry_run_ids(entry, summaries) == ["new", "old"]
    assert entry_run_ids(pinned, summaries) == ["pinned", "new", "old"]


def test_catalogue_without_runs_has_a_pending_row_per_recipe() -> None:
    payload = build_catalogue([], {}, None)

    assert payload["steps"] == STEPS
    assert [e["name"] for e in payload["entries"]] == [e.name for e in CATALOGUE]
    assert len(payload["rows"]) == len(CATALOGUE)
    for entry, row in zip(CATALOGUE, payload["rows"], strict=True):
        assert row["name"] == entry.name and row["run"] is None
        assert row["context_length"] == context_length(recipe_defaults(entry.recipe))


def test_run_fields_falls_back_to_the_recipe_gpu_type(monkeypatch) -> None:
    # Slime run records serialize the recipe's CLI flags, which omit gpu_type.
    record = {
        "training_run_id": "run-1",
        "created_at": 1700000000,
        "status": "completed",
        "config": {
            "model": {"model_name": "zai-org/GLM-5.3-Flash"},
            "recipe": {
                "actor_num_nodes": 3,
                "actor_num_gpus_per_node": 8,
                "colocate": True,
                "rollout_max_context_len": 32768,
            },
            "dataset": {"hf_repo": "SWE-bench/SWE-smith"},
        },
    }
    monkeypatch.setattr(catalogue, "vol_get", lambda store, run_id: record)
    monkeypatch.setattr(
        catalogue,
        "vol_list",
        lambda store: [
            {
                "steps": {
                    "1": {REWARD_KEY: 0.1, "perf/step_time": 600.0},
                    "2": {REWARD_KEY: 0.2, "perf/step_time": 660.0},
                    "3": {REWARD_KEY: 0.3, "perf/step_time": 630.0},
                }
            }
        ],
    )
    monkeypatch.setattr(catalogue, "measured_run_times", lambda rid: ({}, {}))
    entry = next(e for e in CATALOGUE if e.recipe == "GLM_5_3_Flash_LoRA_Recipe")

    fields = run_fields(entry, "run-1", {"gpu_hour_cost_h200": "4.0"}, None)

    run = fields["run"]
    assert fields["context_length"] == 32768
    assert run["gpu_type"] == "H200"
    assert (run["gpus"], run["nodes"]) == (24, 3)
    assert run["gpu_hour_usd"] == 4.0
    assert run["time_per_step_s"] == 630.0
    assert run["cost_per_step_usd"] == round(24 * 4.0 * 630 / 3600, 2)
    assert run["initial_reward"] == 0.1
    assert run["dashboard_url"] == "/training/run-1"


def test_sweep_entries_adds_registry_models_autoconfig_swept() -> None:
    from modal_dojo.common.catalogue import (
        AUTOCONFIG_RECIPE_TAG,
        CatalogueEntry,
        sweep_entries,
    )

    def tagged(name: str, recipe: str | None) -> RunSummary:
        overrides = {AUTOCONFIG_ENTRY_TAG: name}
        if recipe:
            overrides[AUTOCONFIG_RECIPE_TAG] = recipe
        return RunSummary(
            training_run_id=f"run-{name}",
            run_id=f"run-{name}",
            created_at=1,
            group_id="autoconfig-plan-1",
            group_tags=GroupTags(group_id="autoconfig-plan-1", overrides=overrides),
        )

    summaries = [
        tagged("Qwen3.6-27B", "Qwen3_6_27B_Recipe"),
        tagged("Qwen3.6-27B", "Qwen3_6_27B_Recipe"),
        tagged("Qwen3.8-27B", "Qwen3_8_27B_Recipe"),  # already in CATALOGUE
        tagged("Qwen3-4B", None),  # recipe unknown: cannot fill defaults
        tagged("Nope-1B", "Qwen3_4B_Recipe"),  # not a registry model
    ]
    assert sweep_entries(summaries) == [
        CatalogueEntry("Qwen3.6-27B", "Qwen3_6_27B_Recipe")
    ]
