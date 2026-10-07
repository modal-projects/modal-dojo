from __future__ import annotations

import pytest

import modal_dojo
from modal_dojo.common.models.validation import _ValidationConfig
from scripts import generate_catalogue
from scripts.generate_catalogue import (
    CATALOGUE,
    REWARD_KEY,
    context_length,
    gpu_hour_cost,
    logged_per_step,
    native_context_length,
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


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        # Multimodal models keep the language model's window in a sub-config.
        ({"text_config": {"max_position_embeddings": 262144}}, 262144),
        (
            {"thinker_config": {"text_config": {"max_position_embeddings": 32768}}},
            32768,
        ),
        # YaRN that records its original window already states the long one.
        (
            {
                "max_position_embeddings": 1048576,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                },
            },
            1048576,
        ),
        # Plain scaling stretches the configured window.
        (
            {
                "max_position_embeddings": 32768,
                "rope_scaling": {"rope_type": "linear", "factor": 4},
            },
            131072,
        ),
        # sglang's key order: seq_length comes before max_position_embeddings.
        ({"seq_length": 8192, "max_position_embeddings": 32768}, 8192),
        ({"hidden_size": 4096}, None),
    ],
)
def test_native_context_length_follows_sglang(config, expected) -> None:
    assert native_context_length(config) == expected


def test_recipe_context_settings_win_over_the_model_window(monkeypatch) -> None:
    monkeypatch.setattr(
        generate_catalogue, "model_context_length", lambda model: 262144
    )

    assert context_length({"rollout_max_context_len": 32768}, "org/m") == (
        32768,
        "recipe",
    )
    assert context_length(
        {"extra_config": {"rollout_max_context_len": 65536}}, "org/m"
    ) == (65536, "recipe")
    assert context_length({"sglang_context_length": 16384}, "org/m") == (
        16384,
        "recipe",
    )
    # A response cap alone leaves the server at the model's own window.
    assert context_length({"rollout_max_response_len": 4096}, "org/m") == (
        262144,
        "model",
    )


def test_planned_context_length_follows_the_recipe_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        generate_catalogue, "model_context_length", lambda model: 262144
    )

    glm = recipe_defaults("GLM_5_3_Flash_LoRA_Recipe")
    qwen = recipe_defaults("Qwen3_6_27B_Recipe")
    assert context_length(glm, "zai-org/GLM-5.3-Flash") == (32768, "recipe")
    assert context_length(qwen, "Qwen/Qwen3.6-27B") == (262144, "model")


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
