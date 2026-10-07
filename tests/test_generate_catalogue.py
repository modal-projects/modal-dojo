from __future__ import annotations

import pytest

import modal_dojo
from modal_dojo.common.models.validation import _ValidationConfig
from scripts.generate_catalogue import (
    CATALOGUE,
    REWARD_KEY,
    context_length,
    gpu_hour_cost,
    logged_per_step,
    recipe_defaults,
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
