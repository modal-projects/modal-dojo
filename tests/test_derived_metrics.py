"""Default per-step metrics: tokens, throughput and cost, derived at ingest."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo.common import derived_metrics as dm
from modal_dojo.common.derived_metrics import (
    RunPricing,
    changed_points,
    derived_points,
    driver_step_seconds,
    gpu_hour_cost,
    samples_per_step,
)
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun, TrainingRunStatus
from modal_dojo.utils import metadata
from modal_dojo.utils.metadata import MetadataStore

RUN_ID = "priced-run"
AUTH = {"Authorization": "Bearer secret"}
RATES = {"gpu_hour_cost_h200": "4.0", "gpu_hour_cost_a100_80gb": "2.5"}
# 8 H200 at $4/h: $8.888/s... keep the arithmetic round: 8 * 4 / 3600 USD per second.
USD_PER_S = 8 * 4.0 / 3600


@pytest.mark.parametrize(
    ("gpu_type", "expected"),
    [("H200", 4.0), ("H200:8", 4.0), ("A100-80GB!", 2.5), ("B300", None)],
)
def test_gpu_hour_cost_matches_billing_rate_keys(gpu_type, expected) -> None:
    assert gpu_hour_cost(RATES, gpu_type) == expected


def test_samples_per_step_is_prompts_times_samples() -> None:
    assert samples_per_step({"rollout_batch_size": 16, "n_samples_per_prompt": 4}) == 64
    assert samples_per_step({"rollout_batch_size": 16}) == 16
    assert samples_per_step({}) is None


def test_pricing_comes_from_the_run_recipe_and_rates() -> None:
    recipe = {
        "gpu_type": "H200",
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 8,
        "colocate": True,
        "rollout_num_gpus_per_engine": 2,
        "rollout_batch_size": 8,
        "n_samples_per_prompt": 2,
    }
    pricing = RunPricing.from_recipe(recipe, RATES)
    assert (pricing.gpus, pricing.gpu_hour_usd, pricing.samples_per_step) == (
        8,
        4.0,
        16,
    )
    assert pricing.usd_per_s == pytest.approx(USD_PER_S)
    unpriced = RunPricing.from_recipe({**recipe, "gpu_type": "B300"}, RATES)
    assert unpriced.gpu_hour_usd is None and unpriced.usd_per_s is None
    # A recipe the allocator can't resolve still counts tokens, just unpriced.
    unresolved = RunPricing.from_recipe(
        {"gpu_type": "H200", "rollout_batch_size": 8}, RATES
    )
    assert unresolved.gpus == 0 and unresolved.usd_per_s is None


def lanes(*rollouts: tuple[int, dict[str, float], bool]) -> dict:
    """Driver lanes: (rollout id, phase busy seconds, has train_models)."""
    out = {}
    for rollout_id, busy, trained in rollouts:
        phases = {name: {"busy_duration_s": s} for name, s in busy.items()}
        if trained:
            phases["train_models"] = {"busy_duration_s": 10.0}
        out[str(rollout_id)] = {"roles": {"driver": {"phases": phases}}}
    return out


def test_driver_step_seconds_skip_checkpoints_evals_and_unfinished_steps() -> None:
    measured = driver_step_seconds(
        lanes(
            (0, {"generate_rollouts": 100.0, "checkpoint_save": 50.0}, True),
            (1, {"generate_rollouts": 200.0, "evaluate_rollouts": 9.0}, True),
            (2, {"generate_rollouts": 300.0}, False),
        )
    )
    assert measured == {1: 110.0, 2: 210.0}


PRICING = RunPricing(gpus=8, gpu_hour_usd=4.0, samples_per_step=10)


def test_derived_points_price_each_logged_step() -> None:
    table = {
        5: {"rollout/response_len/mean": 100.0, "perf/step_time": 50.0},
        7: {"perf/actor_train_tok_per_s": 1000.0, "perf/actor_train_time": 2.0},
        9: {"rollout/response_len/mean": 200.0, "perf/step_time": 100.0},
    }
    points = derived_points(table, {1: 40.0}, PRICING)
    # Step 1: 1000 generated tokens (row 5) + 2000 trained (row 7) over the
    # measured 40 s; step 2 falls back to perf/step_time.
    assert points[5]["dojo/gen_tokens"] == 1000.0
    assert points[7]["dojo/train_tokens"] == 2000.0
    assert points[5]["dojo/tokens"] == 3000.0
    assert points[5]["dojo/step_seconds"] == 40.0
    assert points[5]["dojo/tokens_per_s"] == pytest.approx(75.0)
    assert points[5]["dojo/usd_per_mtok"] == round(USD_PER_S * 40 / 3000 * 1e6, 4)
    assert points[5]["dojo/step_cost_usd"] == round(USD_PER_S * 40, 4)
    assert points[5]["dojo/gpu_hour_usd"] == 4.0
    assert points[9]["dojo/step_seconds"] == 100.0
    assert points[9]["dojo/gen_tokens_per_s"] == pytest.approx(20.0)
    assert "dojo/train_tokens" not in points[9]


def test_derived_points_leave_out_what_cannot_be_priced() -> None:
    table = {1: {"rollout/response_len/mean": 100.0, "perf/step_time": 50.0}}
    unpriced = derived_points(table, {}, RunPricing(8, None, 10))
    assert "dojo/tokens_per_s" in unpriced[1] and "dojo/usd_per_mtok" not in unpriced[1]
    unsampled = derived_points(table, {}, RunPricing(8, 4.0, None))
    assert (
        "dojo/gen_tokens" not in unsampled[1] and "dojo/step_cost_usd" in unsampled[1]
    )
    assert derived_points({1: {"loss": 0.1}}, {}, PRICING) == {}


def test_changed_points_only_reports_new_or_different_values() -> None:
    table = {1: {"dojo/tokens": 3.0, "dojo/step_seconds": 1.0}}
    assert (
        changed_points(table, {1: {"dojo/tokens": 3.0, "dojo/step_seconds": 1.0}}) == {}
    )
    assert changed_points(
        table, {1: {"dojo/tokens": 4.0, "dojo/step_seconds": 1.0}}
    ) == {1: {"dojo/tokens": 4.0}}


# ── ingest ────────────────────────────────────────────────────────────────


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr(dm, "billing_rates", lambda: RATES)
    monkeypatch.setattr(_dashboard, "billing_rates", lambda: RATES)
    return TestClient(_dashboard.fastapi_app.local())


def _save_run() -> None:
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-priced",
        framework=Framework.SLIME,
        config={
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "recipe": {
                "gpu_type": "H200",
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 8,
                "colocate": True,
                "rollout_num_gpus_per_engine": 2,
                "rollout_batch_size": 10,
            },
        },
        created_at=100,
        started_at=100,
        updated_at=150,
        status=TrainingRunStatus.RUNNING,
    ).save()
    metadata.vol_put(MetadataStore.FRAMEWORK_STATUS_TOKENS, RUN_ID, {"token": "secret"})


def _timing(rollout_id: int, busy: float) -> dict:
    phase = {
        "count": 1,
        "busy_duration_s": busy,
        "longest_invocation_s": busy,
        "first_start_s": 0.0,
        "last_end_s": busy,
    }
    return {
        "training_run_id": RUN_ID,
        "rollout_id": rollout_id,
        "role": "driver",
        "lane_start_unix_s": 100.0,
        "final": True,
        "phases": {
            "generate_rollouts": phase,
            "train_models": {**phase, "busy_duration_s": 0.0},
        },
    }


def test_ingest_adds_priced_dojo_metrics_to_the_series(
    fake_volume, monkeypatch, tmp_path
):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        # Timing for step 1 arrives before its metrics.
        assert (
            client.post(
                "/api/timing-events", json=_timing(0, 40.0), headers=AUTH
            ).status_code
            == 200
        )
        batch = {
            "training_run_id": RUN_ID,
            "points": [
                {
                    "step": 1,
                    "metrics": {
                        "rollout/response_len/mean": 100.0,
                        "perf/step_time": 99.0,
                    },
                },
            ],
            "final": True,
        }
        assert (
            client.post("/api/metric-points", json=batch, headers=AUTH).status_code
            == 200
        )
        series = client.get(f"/api/runs/{RUN_ID}/metrics").json()["series"]
    assert series["dojo/step_seconds"][0][1] == 40.0  # measured beats perf/step_time
    assert series["dojo/gen_tokens"][0][1] == 1000.0
    assert series["dojo/tokens_per_s"][0][1] == pytest.approx(25.0)
    assert series["dojo/usd_per_mtok"][0][1] == round(USD_PER_S * 40 / 1000 * 1e6, 4)
    assert series["dojo/step_cost_usd"][0][1] == round(USD_PER_S * 40, 4)


def test_ingest_prices_later_when_timing_arrives_after_metrics(
    fake_volume, monkeypatch, tmp_path
):
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        batch = {
            "training_run_id": RUN_ID,
            "points": [{"step": 1, "metrics": {"rollout/response_len/mean": 100.0}}],
            "final": False,
        }
        assert (
            client.post("/api/metric-points", json=batch, headers=AUTH).status_code
            == 200
        )
        before = client.get(f"/api/runs/{RUN_ID}/metrics").json()["series"]
        assert "dojo/gen_tokens" in before and "dojo/tokens_per_s" not in before
        assert (
            client.post(
                "/api/timing-events", json=_timing(0, 20.0), headers=AUTH
            ).status_code
            == 200
        )
        after = client.get(f"/api/runs/{RUN_ID}/metrics").json()["series"]
    assert after["dojo/step_seconds"][0][1] == 20.0
    assert after["dojo/tokens_per_s"][0][1] == pytest.approx(50.0)


def test_reading_a_finished_run_backfills_and_persists_dojo_metrics(
    fake_volume, monkeypatch, tmp_path
):
    """Runs logged before derived metrics existed get them on first read."""
    from modal_dojo.common.metric_series import metric_series_store
    from modal_dojo.common.step_timing import RoleTimingRecord

    _save_run()
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-priced",
        framework=Framework.SLIME,
        config={
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "recipe": {
                "gpu_type": "H200",
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 8,
                "colocate": True,
                "rollout_num_gpus_per_engine": 2,
                "rollout_batch_size": 10,
            },
        },
        created_at=100,
        started_at=100,
        updated_at=150,
        status=TrainingRunStatus.COMPLETED,
    ).save()
    metadata.vol_put(
        metric_series_store(RUN_ID),
        "chunk-000000-old",
        {
            "steps": {"1": {"rollout/response_len/mean": 100.0}},
            "written": {"1": {"rollout/response_len/mean": 1.0}},
        },
    )
    RoleTimingRecord.model_validate(_timing(0, 40.0)).save()
    with _client(monkeypatch, tmp_path) as client:
        series = client.get(f"/api/runs/{RUN_ID}/metrics").json()["series"]
    assert series["dojo/gen_tokens"][0][1] == 1000.0
    assert series["dojo/tokens_per_s"][0][1] == pytest.approx(25.0)
    persisted = [
        path
        for path in fake_volume.files
        if path.startswith(f"{metric_series_store(RUN_ID)}/")
    ]
    assert any(
        path.endswith(f"-{_dashboard.METRIC_WRITER_ID}.json") for path in persisted
    )
