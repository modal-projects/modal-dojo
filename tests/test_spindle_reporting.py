"""SpindleRecipe's Dojo reporting queues."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from modal_dojo.common.status import SpindleStatus
from modal_dojo.frameworks.spindle.reporting import DojoReporter

CONFIG = SimpleNamespace(
    model_name="Qwen/Qwen3.5-2B",
    learning_rate=1e-5,
    lora_rank=32,
    loss_fn="importance_sampling",
    max_tokens=64,
    max_steps=10,
    recipe_name="test",
    renderer_name="qwen3_5_disable_thinking",
    dataset_builder=object(),
    log_path="/tmp/dojo-spindle-test",
)


def test_pending_status_updates_merge_into_latest_snapshot():
    async def pending_status():
        reporter = DojoReporter(CONFIG, {})
        reporter._phase(SpindleStatus.TRAIN_STEP, is_active=True, progress_current=3)
        reporter._phase(SpindleStatus.SAVE_CHECKPOINT_AND_GET_SAMPLING_CLIENT)
        queue = reporter.queues["framework-status"]
        return queue.qsize(), queue.get_nowait()

    size, payload = asyncio.run(pending_status())
    assert size == 1
    assert payload["phase"] == "save_checkpoint_and_get_sampling_client"
    assert payload["progress_current"] == 3


def test_other_streams_keep_every_payload():
    async def pending_metrics():
        reporter = DojoReporter(CONFIG, {})
        reporter._enqueue("metric-points", {"points": [1]})
        reporter._enqueue("metric-points", {"points": [2]})
        return reporter.queues["metric-points"].qsize()

    assert asyncio.run(pending_metrics()) == 2
