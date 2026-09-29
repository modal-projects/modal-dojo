"""Drive the real ``save()`` chain to completion without Modal or a GPU.

``TrainingRun.save()``, ``save_train_result_blob()``, and
``TrainingRolloutResult.save(is_async=True)`` are exercised against an in-memory
``FakeVolume`` (see ``conftest.py``) so the full serialize-and-write path is
covered in CI: the payload must stay JSON-serializable, and ``save()`` (in both
sync and ``is_async=True`` modes) must complete even when ``Volume.reload()``
is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import nullcontext
from unittest.mock import AsyncMock, Mock

import pytest
from modal.exception import ExecutionError

from modal_dojo.common import run as run_mod
from modal_dojo.common.framework import Framework
from modal_dojo.common import launcher_helpers
from modal_dojo.common.launcher_helpers import (
    complete_training_run,
    training_run_lifecycle,
)
from modal_dojo.common.train_result import (
    save_train_result_blob,
    train_result_payload,
)
from modal_dojo.common.training_rollout import TrainingRolloutResult
from modal_dojo.utils import metadata
from modal_dojo.utils.metadata import MetadataStore


@pytest.mark.parametrize(
    "error, expected",
    [
        (None, "completed"),
        (RuntimeError("worker failed"), "failed"),
        (KeyboardInterrupt(), "stopped"),
    ],
)
def test_training_lifecycle_persists_terminal_state(
    fake_volume, monkeypatch, error, expected
):
    metrics = {"provider": "wandb", "project": "test", "run_id": "metrics-run"}
    model_config = {"model_name": "Qwen/Qwen3-4B", "model_path": None}
    run = run_mod.TrainingRun(
        training_run_id="lifecycle",
        framework=Framework.SLIME,
        config={"metrics": metrics},
    )
    run_mod.set_checkpoint_location(
        run,
        checkpoint_dir="/checkpoints/run",
        checkpoints_volume_name="outputs",
        checkpoints_mount_path="/checkpoints",
    )
    run.save()

    async def exercise():
        async with training_run_lifecycle(run):
            pass
        assert run_mod.TrainingRun.from_id("lifecycle").ended_at is None
        async with training_run_lifecycle(run):
            latest = await run_mod.TrainingRun.from_id("lifecycle", is_async=True)
            latest.metadata["dashboard_update"] = "preserved"
            await latest.save(is_async=True)
            if error is not None:
                raise error
            payload = await complete_training_run(
                run,
                checkpoints_volume=Mock(commit=Mock(aio=AsyncMock())),
                app_name="test",
                model=model_config,
                group_id=None,
            )
            assert payload["checkpoint_dir"] == "/checkpoints/run"
            assert payload["checkpoints_volume_name"] == "outputs"

    with pytest.raises(type(error)) if error else nullcontext():
        asyncio.run(exercise())
    saved = run_mod.TrainingRun.from_id("lifecycle")
    assert saved.status.value == saved.metadata["last_attempt_status"] == expected
    assert saved.ended_at is not None
    assert saved.metadata["dashboard_update"] == "preserved"
    if error is None:
        assert saved.app_name == "test"
        assert saved.source_model == model_config
        assert saved.metrics == metrics
        monkeypatch.setattr(run_mod.TrainingRun, "latest_checkpoint", lambda self: None)
        model = saved.model
        assert model.model_name == model_config["model_name"]
        assert model.model_path == "/checkpoints/run"
    if isinstance(error, RuntimeError):
        assert "worker failed" in saved.error_message


@pytest.mark.parametrize("fw", list(Framework))
def test_training_run_save_survives_unmounted_volume(fake_volume, fw):
    """TrainingRun.save() completes when reload() raises, for every framework."""
    run_mod.TrainingRun(training_run_id="t1", framework=fw, config={}).save()

    blob = fake_volume.files[f"{MetadataStore.TRAINING_RUNS.value}/t1.json"]
    assert json.loads(blob)["framework"] == fw.value


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_blob_save_survives_unmounted_volume(fake_volume, fw):
    """``save_train_result_blob()`` completes when reload() raises, for every framework."""
    save_train_result_blob(
        train_result_payload(app_name="a", framework=fw, training_run_id="t2")
    )

    blob = fake_volume.files[f"{MetadataStore.TRAIN_RESULTS.value}/t2.json"]
    assert json.loads(blob)["framework"] == fw.value


def test_rollout_async_save_survives_unmounted_volume(fake_volume):
    """TrainingRolloutResult.save(is_async=True) completes when reload() raises."""
    asyncio.run(
        TrainingRolloutResult(
            training_run_id="t3",
            rollout_id=0,
            samples=[{"score": 1.0, "prompt": "p", "response": "r"}],
        ).save(is_async=True)
    )

    blob = fake_volume.files[
        f"{MetadataStore.TRAINING_ROLLOUTS.value}/t3__00000000.json"
    ]
    assert json.loads(blob)["rollout_id"] == 0
    summary = fake_volume.files[
        f"{MetadataStore.TRAINING_ROLLOUTS_SUMMARY.value}/summary.json"
    ]
    summary_item = json.loads(summary)["items"][0]
    assert summary_item["summary_key"] == "t3__00000000"
    assert summary_item["export_size_bytes"] == len(
        (json.dumps(json.loads(blob), ensure_ascii=False, indent=2) + "\n").encode()
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_summary_upsert_survives_unreadable_summary_file(
    fake_volume, monkeypatch, is_async
):
    """Summary upserts rebuild from canonical items when the cache is unreadable."""
    summary_path = (
        f"{metadata._store_path(MetadataStore.TRAINING_RUNS_SUMMARY)}/summary.json"
    )
    original_read = fake_volume._read_file
    original_read_async = fake_volume._read_file_async

    def unreadable_read(path: str):
        if path == summary_path:
            from modal.exception import ExecutionError

            raise ExecutionError(
                "Request failed with status 404 Not Found: block not found"
            )
        return original_read(path)

    async def unreadable_read_async(path: str):
        if path == summary_path:
            from modal.exception import ExecutionError

            raise ExecutionError(
                "Request failed with status 404 Not Found: block not found"
            )
        async for chunk in original_read_async(path):
            yield chunk

    def put(*args, **kwargs):
        return metadata.vol_put(*args, is_async=is_async, **kwargs)

    def put_with_summary(*args, **kwargs):
        return metadata.vol_put_with_summary(*args, is_async=is_async, **kwargs)

    if is_async:
        asyncio.run(
            put(
                MetadataStore.TRAINING_RUNS,
                "run-a",
                {"training_run_id": "run-a"},
            )
        )
    else:
        put(
            MetadataStore.TRAINING_RUNS,
            "run-a",
            {"training_run_id": "run-a"},
        )

    with monkeypatch.context() as summary_monkeypatch:
        summary_monkeypatch.setattr(fake_volume, "_read_file", unreadable_read)
        summary_monkeypatch.setattr(
            fake_volume, "_read_file_async", unreadable_read_async
        )
        summary_monkeypatch.setattr(fake_volume.read_file, "_sync_fn", unreadable_read)
        summary_monkeypatch.setattr(fake_volume.read_file, "aio", unreadable_read_async)

        write = put_with_summary(
            MetadataStore.TRAINING_RUNS,
            "run-b",
            {"training_run_id": "run-b"},
            summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
            item_id_key="training_run_id",
        )
        if is_async:
            asyncio.run(write)

    if is_async:
        items = asyncio.run(
            metadata.vol_get_summary_items(
                MetadataStore.TRAINING_RUNS_SUMMARY, is_async=True
            )
        )
    else:
        items = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert {item["training_run_id"] for item in items or []} == {"run-a", "run-b"}


@pytest.mark.parametrize("fw", list(Framework))
def test_train_result_payload_is_json_serializable(fw):
    payload = train_result_payload(app_name="a", framework=fw, training_run_id="t")
    assert json.loads(json.dumps(payload))["framework"] == fw.value


@pytest.mark.skipif(
    os.environ.get("RUN_MODAL_TESTS") != "1",
    reason="hits Modal (no GPU); opt in with RUN_MODAL_TESTS=1",
)
def test_remote_save_from_unmounted_container():
    """The faithful remote counterpart: run save() inside a real Modal container
    that does *not* mount the metadata volume — the exact context where the
    original training run crashed with `volume … not attached`. The fake-volume
    tests simulate that; this proves it against real Modal Volume semantics
    (reload unavailable, but the client-side write still lands).
    """
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("modal>=1.4.0", "pydantic")
        .add_local_python_source("modal_dojo")
    )
    app = modal.App("training-gym-metadata-save-probe")

    # NB: deliberately no volumes= — this is the unmounted case.
    @app.function(image=image, serialized=True)
    def _save_probe() -> str:
        from modal_dojo.common.framework import Framework
        from modal_dojo.common.run import TrainingRun
        from modal_dojo.common.train_result import (
            save_train_result_blob,
            train_result_payload,
        )

        # Framework is incidental here — this probes volume/reload mechanics,
        # which are framework-independent. Per-framework serialization is
        # covered by the fast parametrized tests above, so we don't pay for N
        # real Modal apps to re-check it.
        rid = "ci-remote-save-probe"  # fixed id → overwrites, no junk accrual
        TrainingRun(training_run_id=rid, framework=Framework.SLIME, config={}).save()
        save_train_result_blob(
            train_result_payload(
                app_name=rid, framework=Framework.SLIME, training_run_id=rid
            )
        )
        return "ok"

    with modal.enable_output():
        with app.run():
            assert _save_probe.remote() == "ok"


def test_terminal_save_failure_preserves_training_error(monkeypatch, fake_volume):
    record = run_mod.TrainingRun(training_run_id="run", framework="slime", config={})
    monkeypatch.setattr(
        launcher_helpers,
        "build_terminal_run_record",
        AsyncMock(return_value=Mock(save=AsyncMock(side_effect=RuntimeError("io")))),
    )
    original = RuntimeError("training failed")

    async def run():
        with pytest.raises(RuntimeError) as caught:
            async with training_run_lifecycle(record):
                raise original
        assert caught.value is original

    asyncio.run(run())


def test_sync_list_reads_files_concurrently(fake_volume, monkeypatch):
    for i in range(2):
        metadata.vol_put(
            MetadataStore.TRAINING_RUNS, f"run-{i}", {"training_run_id": f"run-{i}"}
        )
    both_reading = threading.Barrier(2, timeout=5)
    read_file = fake_volume.read_file

    def read_when_both_reading(path: str):
        both_reading.wait()
        return read_file(path)

    monkeypatch.setattr(fake_volume, "read_file", read_when_both_reading)

    records = metadata.vol_list(MetadataStore.TRAINING_RUNS)

    assert sorted(r["training_run_id"] for r in records) == ["run-0", "run-1"]


def test_compaction_writes_readable_records_before_raising(fake_volume, monkeypatch):
    metadata.vol_put_summary_items(
        MetadataStore.TRAINING_RUNS_SUMMARY,
        [{"training_run_id": f"run-{i}", "status": "running"} for i in range(3)],
    )
    for i in range(3):
        metadata.vol_put(
            MetadataStore.TRAINING_RUNS,
            f"run-{i}",
            {"training_run_id": f"run-{i}", "status": "completed"},
        )
    unreadable = f"{metadata._store_path(MetadataStore.TRAINING_RUNS)}/run-1.json"
    read_file = fake_volume.read_file

    def read_or_fail(path: str):
        if path == unreadable:
            raise ExecutionError("block not found")
        return read_file(path)

    monkeypatch.setattr(fake_volume, "read_file", read_or_fail)

    with pytest.raises(ExecutionError):
        metadata.vol_compact_summary_items(
            MetadataStore.TRAINING_RUNS_SUMMARY,
            MetadataStore.TRAINING_RUNS,
            item_id_key="training_run_id",
        )

    items = metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY)
    assert {item["training_run_id"]: item["status"] for item in items} == {
        "run-0": "completed",
        "run-1": "running",
        "run-2": "completed",
    }


def test_compaction_keeps_unread_summary_when_canonical_read_fails(
    fake_volume, monkeypatch
):
    summary = [{"training_run_id": f"run-{i}", "status": "running"} for i in range(2)]
    metadata.vol_put_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY, summary)
    for i in range(2):
        metadata.vol_put(
            MetadataStore.TRAINING_RUNS,
            f"run-{i}",
            {"training_run_id": f"run-{i}", "status": "completed"},
        )
    unreadable = {
        f"{metadata._store_path(MetadataStore.TRAINING_RUNS_SUMMARY)}/{metadata.SUMMARY_KEY}.json",
        f"{metadata._store_path(MetadataStore.TRAINING_RUNS)}/run-1.json",
    }
    read_file = fake_volume.read_file

    def read_or_fail(path: str):
        if path in unreadable:
            raise ExecutionError("block not found")
        return read_file(path)

    monkeypatch.setattr(fake_volume, "read_file", read_or_fail)

    with pytest.raises(ExecutionError):
        metadata.vol_compact_summary_items(
            MetadataStore.TRAINING_RUNS_SUMMARY,
            MetadataStore.TRAINING_RUNS,
            item_id_key="training_run_id",
        )

    monkeypatch.setattr(fake_volume, "read_file", read_file)
    assert (
        metadata.vol_get_summary_items(MetadataStore.TRAINING_RUNS_SUMMARY) == summary
    )
