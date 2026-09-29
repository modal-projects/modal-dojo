from __future__ import annotations

import json
import time
from queue import Queue
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from modal_dojo import _dashboard
from modal_dojo.common import reporting
from modal_dojo.common.framework import Framework
from modal_dojo.common.run import TrainingRun
from modal_dojo.common.step_timing import RoleTimingRecord
from modal_dojo.common.train_result import (
    save_train_result_blob,
    train_result_payload,
)
from modal_dojo.common.training_rollout import TrainingRolloutResult
from modal_dojo.utils import metadata
from modal_dojo.utils.metadata import MetadataStore


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    (static / "apple-touch-icon.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    )
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_records() -> None:
    TrainingRun(
        training_run_id="run-route-1",
        modal_app_id="ap-route",
        framework=Framework.SLIME,
        config={
            "model": {"model_name": "Qwen/Qwen3-4B"},
            "dataset": {"hf_repo": "openai/gsm8k"},
            "recipe": {"gpu_type": "H100"},
        },
        created_at=100,
        started_at=100,
        updated_at=150,
        metadata={"group_id": "route-group"},
    ).save()
    save_train_result_blob(
        train_result_payload(
            app_name="route-app",
            framework=Framework.SLIME,
            training_run_id="run-route-1",
            checkpoint_dir="/checkpoints/run-route-1",
        )
    )


def test_runs_route_returns_typed_joined_summaries(fake_volume, monkeypatch, tmp_path):
    _save_records()

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    summary = response.json()[0]
    assert summary["training_run_id"] == "run-route-1"
    assert summary["run_id"] == "run-route-1"
    assert summary["status"] == "running"
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["dataset"] == "openai/gsm8k"
    assert summary["recipe"] == "slime"
    assert summary["group_id"] == "route-group"
    assert summary["display_status"] == "completed"
    assert summary["display_stage"] == ""
    assert "list_fields" not in summary
    assert summary["has_train_result"] is True
    assert summary["train_result"]["checkpoint_dir"] == ("/checkpoints/run-route-1")


def test_get_run_route_returns_one_typed_summary(fake_volume, monkeypatch, tmp_path):
    _save_records()

    def fail_summary_scan(_store):
        raise AssertionError("single-run route must not scan summary stores")

    monkeypatch.setattr(metadata, "vol_get_summary_items_healed", fail_summary_scan)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/run-route-1")

    assert response.status_code == 200
    summary = response.json()
    assert summary["training_run_id"] == "run-route-1"
    assert summary["model"] == "Qwen/Qwen3-4B"
    assert summary["display_status"] == "completed"
    assert summary["train_result"]["checkpoint_dir"] == "/checkpoints/run-route-1"


def test_get_run_route_returns_404_for_unknown_run(fake_volume, monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Training run 'missing' not found"


def test_rollout_route_preserves_raw_text_and_adds_cleaned_text(
    fake_volume, monkeypatch, tmp_path
):
    raw_prompt = "<|im_start|>user\nraw prompt<|im_end|>"
    raw_response = "<think>secret</think>raw answer"
    TrainingRolloutResult(
        training_run_id="run-route-1",
        rollout_id=3,
        samples=[
            {
                "score": 0.5,
                "prompt": raw_prompt,
                "response": raw_response,
                "parsed_response": {
                    "content": "raw answer",
                    "thinking": "secret",
                },
            }
        ],
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        display = client.get("/api/runs/run-route-1/rollouts/3")

    assert display.status_code == 200
    display_sample = display.json()["samples"][0]
    assert display_sample["prompt"] == "raw prompt"
    assert display_sample["response"] == "raw answer"
    assert display_sample["raw_prompt"] == raw_prompt
    assert display_sample["raw_response"] == raw_response
    assert display_sample["parsed_response"]["thinking"] == "secret"


def test_runs_route_keeps_runs_when_train_result_store_fails(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    original = metadata.vol_get_summary_items_healed

    def fail_results(store):
        if store is MetadataStore.TRAIN_RESULTS_SUMMARY:
            raise RuntimeError("result store unavailable")
        return original(store)

    monkeypatch.setattr(metadata, "vol_get_summary_items_healed", fail_results)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["has_train_result"] is False


def test_runs_route_filters_and_sorts_by_creation(fake_volume, monkeypatch, tmp_path):
    _save_records()
    TrainingRun(
        training_run_id="run-route-2",
        framework=Framework.MILES,
        status="failed",
        config={
            "model": {"model_name": "other/model"},
            "dataset": {"hf_repo": "other/data"},
        },
        created_at=50,
        started_at=50,
        updated_at=2_000_000_000,
        metadata={"group_id": "other-group"},
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        filtered = client.get("/api/runs?display_status=failed&since=175&limit=1")
        all_runs = client.get("/api/runs")

    assert filtered.status_code == 200
    assert [run["run_id"] for run in filtered.json()] == ["run-route-2"]
    # ``since`` still selects on update time, but the list is ordered by
    # creation: sorting by update time reshuffles the pages under a client that
    # is paging through the list whenever a run reports progress.
    assert [run["run_id"] for run in all_runs.json()] == [
        "run-route-1",
        "run-route-2",
    ]


def test_runs_route_pages_and_omits_detail_only_fields(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    TrainingRun(
        training_run_id="run-route-2",
        framework=Framework.MILES,
        status="failed",
        config={
            "model": {"model_name": "other/model"},
            "dataset": {"hf_repo": "other/data"},
        },
        created_at=50,
        started_at=50,
        updated_at=2_000_000_000,
        metadata={"group_id": "other-group"},
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        first = client.get("/api/runs?limit=1")
        second = client.get("/api/runs?limit=1&offset=1")
        searched = client.get("/api/runs?q=OTHER/MODEL")
        faceted = client.get("/api/runs?status=failed")
        detail = client.get("/api/runs/run-route-1")

    assert [run["run_id"] for run in first.json()] == ["run-route-1"]
    assert [run["run_id"] for run in second.json()] == ["run-route-2"]
    assert [run["run_id"] for run in searched.json()] == ["run-route-2"]
    assert [run["run_id"] for run in faceted.json()] == ["run-route-2"]
    # The table renders none of these, and they are most of the payload.
    assert not {"config", "step_times", "substep_times"} & set(first.json()[0])
    assert detail.json()["config"]


def test_runs_route_unions_repeated_facet_values(fake_volume, monkeypatch, tmp_path):
    _save_records()
    TrainingRun(
        training_run_id="run-route-2",
        framework=Framework.MILES,
        config={"model": {"model_name": "other/model"}},
        created_at=50,
        started_at=50,
        updated_at=200,
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        both = client.get("/api/runs?recipe=slime&recipe=miles")

    # ``recipe`` is a filter-chip facet as well as a filterable column, so a
    # multi-select must not narrow to whichever repeated value is read first.
    assert [run["run_id"] for run in both.json()] == ["run-route-1", "run-route-2"]


def test_runs_counts_route_counts_every_run_and_the_current_query(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    TrainingRun(
        training_run_id="run-route-2",
        framework=Framework.MILES,
        status="failed",
        config={"model": {"model_name": "other/model"}},
        created_at=50,
        started_at=50,
        updated_at=2_000_000_000,
    ).save()

    with _client(monkeypatch, tmp_path) as client:
        counts = client.get("/api/runs/counts").json()
        failed = client.get("/api/runs/counts?status=failed").json()

    assert counts["total"] == 2
    assert counts["matching"] == 2
    assert counts["status"]["failed"] == 1
    assert counts["group"] == {"route-group": 1, "(no group)": 1}
    # Bucket counts always describe every run — they are what the chips would
    # select — while ``matching`` follows the current selection.
    assert failed["status"] == counts["status"]
    assert failed["matching"] == 1


def test_unknown_api_path_is_a_json_404_not_the_spa(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        missing = client.get("/api/nope")
        spa = client.get("/nope")

    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("application/json")
    assert spa.status_code == 200
    assert spa.text == "ok"


def test_apple_touch_icon_is_served_as_png(monkeypatch, tmp_path):
    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/apple-touch-icon.png")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content.startswith(b"\x89PNG")


def test_runs_route_isolates_invalid_run_records(fake_volume, monkeypatch, tmp_path):
    _save_records()
    runs = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert runs is not None
    invalid_run = {
        **runs[0],
        "training_run_id": "invalid-run",
        "step_times": {"1": {"phase": {"not": "an integer"}}},
    }
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_RUNS_SUMMARY, [invalid_run, runs[0]]
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["training_run_id"] for run in response.json()] == ["run-route-1"]


def test_run_log_stream_events_always_include_json_data(
    fake_volume, monkeypatch, tmp_path
):
    _save_records()
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-token-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-token-secret")

    batch = SimpleNamespace(
        entry_id="entry-1",
        task_id="task-1",
        items=[
            SimpleNamespace(data="first\n", timestamp=100.0),
            SimpleNamespace(data="second\n", timestamp=101.0),
        ],
        app_done=True,
    )

    class FakeAppGetLogs:
        def __init__(self):
            self.calls = 0

        def unary_stream(self, _request):
            self.calls += 1

            async def stream():
                if self.calls == 1:
                    raise RuntimeError("temporary failure")
                yield batch

            return stream()

    fake_rpc = FakeAppGetLogs()
    fake_modal_client = SimpleNamespace(
        stub=SimpleNamespace(AppGetLogs=fake_rpc),
    )

    async def from_credentials(*_args, **_kwargs):
        return fake_modal_client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("modal.client._Client.from_credentials", from_credentials)
    monkeypatch.setattr(_dashboard.asyncio, "sleep", no_sleep)

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/run-route-1/logs/stream?max_lines_per_sec=1")

    assert response.status_code == 200
    blocks = [block for block in response.text.split("\n\n") if block]
    assert [block.splitlines()[0] for block in blocks] == [
        "event: reconnect",
        'data: {"task_id": "task-1", "line": "first\\n", "ts": 100.0}',
        "event: dropped",
        "event: done",
    ]
    reconnect_data = blocks[0].splitlines()[1]
    assert json.loads(reconnect_data.removeprefix("data: ")) == {
        "reason": "temporary failure"
    }
    for block in blocks:
        data_line = next(
            line for line in block.splitlines() if line.startswith("data: ")
        )
        assert isinstance(json.loads(data_line.removeprefix("data: ")), dict)


@pytest.mark.parametrize("bound", ["since", "until"])
def test_run_logs_rejects_invalid_time_bound(bound, fake_volume, monkeypatch, tmp_path):
    _save_records()

    with _client(monkeypatch, tmp_path) as client:
        response = client.get(
            "/api/runs/run-route-1/logs",
            params={bound: "yesterday-ish"},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith(
        f"{bound} must be epoch seconds, ISO 8601, or a relative time"
    )


def _timing_payload(run_id: str = "timing-run", *, final: bool = False) -> dict:
    return {
        "training_run_id": run_id,
        "rollout_id": 0,
        "role": "driver",
        "lane_start_unix_s": 100.0,
        "final": final,
        "phases": {
            "train_models": {
                "count": 1,
                "busy_duration_s": 0.1,
                "longest_invocation_s": 0.1,
                "first_start_s": 0.0,
                "last_end_s": 0.1,
            }
        },
    }


def _save_timing_run(run_id: str = "timing-run") -> None:
    TrainingRun(
        training_run_id=run_id,
        modal_app_id="ap-timing",
        framework=Framework.SLIME,
        config={},
    ).save()
    metadata.vol_put(
        MetadataStore.FRAMEWORK_STATUS_TOKENS,
        run_id,
        {"token": "secret"},
    )


def test_final_record_is_written_once_and_not_shadowed_by_progress(
    fake_volume, monkeypatch, tmp_path
):
    _save_timing_run()
    writes = []

    async def capture(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(_dashboard, "_metadata_vol_put_many", capture)
    with _client(monkeypatch, tmp_path) as client:
        assert (
            client.post(
                "/api/timing-events",
                json=_timing_payload(final=True),
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        progress = _timing_payload()
        progress["phases"]["train_models"]["count"] = 99
        assert (
            client.post(
                "/api/timing-events",
                json=progress,
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        timings = client.get("/api/runs/timing-run/timings").json()

    assert len(writes) == 1
    assert timings["0"]["roles"]["driver"]["phases"]["train_models"]["count"] == 1

    queue = Queue()
    delivered = []
    final_payload = {
        "_url": "https://dashboard.test/api/timing-events",
        "training_run_id": "timing-run",
        "storage_key": "00000000__driver",
        "final": True,
    }
    queue.put(None)
    queue.put(None)
    queue.put(final_payload)
    monkeypatch.setattr(reporting, "_REPORT_QUEUE", queue)
    monkeypatch.setattr(reporting, "_post", lambda item: delivered.append(item) or True)
    monkeypatch.setattr(reporting, "_REPORT_DRAIN_DEADLINE", time.monotonic() + 1.0)
    reporting._worker()
    assert delivered == [final_payload]
    assert queue.unfinished_tasks == 0


def test_timing_reads_isolate_malformed_records(fake_volume, monkeypatch, tmp_path):
    _save_timing_run()
    RoleTimingRecord.model_validate(_timing_payload()).save()
    metadata.vol_put(
        RoleTimingRecord.store("timing-run"),
        "00000001__driver",
        {"not": "a timing record"},
    )

    with _client(monkeypatch, tmp_path) as client:
        response = client.get("/api/runs/timing-run/timings")

    assert response.status_code == 200
    assert set(response.json()) == {"0"}


def test_persisted_non_final_timing_survives_stale_listing(
    fake_volume, monkeypatch, tmp_path
):
    _save_timing_run()
    listed = []

    async def list_records(*_args, **_kwargs):
        return (listed[-1] if listed else [], False)

    monkeypatch.setattr(_dashboard, "vol_list_metadata_with_failures", list_records)
    run = TrainingRun.from_id("timing-run")
    run.status = run.status.COMPLETED
    run.save()
    writes = []

    async def capture(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(_dashboard, "_metadata_vol_put_many", capture)
    with _client(monkeypatch, tmp_path) as client:
        progress = _timing_payload()
        assert (
            client.post(
                "/api/timing-events",
                json=progress,
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/framework-status",
                json={"training_run_id": "timing-run", "phase": "training"},
                headers={"Authorization": "Bearer secret"},
            ).status_code
            == 200
        )
        first = client.get("/api/runs/timing-run/timings")
        assert first.status_code == 200
        assert (
            first.json()["0"]["roles"]["driver"]["phases"]["train_models"][
                "first_start_s"
            ]
            == 0.0
        )

    listed.append(
        [
            {
                "path": "substep-timing/timing-run/00000000__driver.json",
                "mtime": 1,
                "size": 2,
            }
        ]
    )
    replacement = {
        **progress,
        "lane_start_unix_s": 200.0,
        "final": True,
    }

    original_read = _dashboard._metadata_vol_get

    async def read_record(store, key, *args, **kwargs):
        if key == "00000000__driver":
            return replacement
        return await original_read(store, key, *args, **kwargs)

    monkeypatch.setattr(_dashboard, "_metadata_vol_get", read_record)
    with _client(monkeypatch, tmp_path / "second") as client:
        second = client.get("/api/runs/timing-run/timings")

    assert writes
    assert second.status_code == 200
    assert second.json()["0"]["roles"]["driver"]["lane_start_unix_s"] == 200.0


def test_spa_html_revalidates_and_assets_are_immutable(monkeypatch, tmp_path):
    """A deploy must never leave a browser holding HTML for a deleted bundle.

    Vite's bundle names are content-hashed, so the previous deploy's
    ``/assets/index-<hash>.js`` is gone the moment a new image ships. If a
    cached ``index.html`` were served without revalidating, it would keep
    requesting that bundle and the page would fail to boot on a 404.
    """
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<html></html>")
    (static / "assets" / "index-abc123.js").write_text("console.log(1)")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)

    with TestClient(_dashboard.fastapi_app.local()) as client:
        page = client.get("/training/some-run")
        assert page.status_code == 200
        assert "no-cache" in page.headers["cache-control"]

        asset = client.get("/assets/index-abc123.js")
        assert asset.status_code == 200
        assert "immutable" in asset.headers["cache-control"]

        # A bundle from an older deploy is a plain 404, not the SPA fallback:
        # the browser must see the failure rather than parse HTML as JavaScript.
        missing = client.get("/assets/index-gone.js")
        assert missing.status_code == 404
