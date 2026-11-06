"""SpindleRecipe's Dojo reporting queues."""

from __future__ import annotations

import asyncio
import re
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
    load_checkpoint_path=None,
)


def test_pending_status_updates_merge_into_latest_snapshot():
    async def pending_status():
        reporter = DojoReporter(CONFIG)
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
        reporter = DojoReporter(CONFIG)
        reporter._enqueue("metric-points", {"points": [1]})
        reporter._enqueue("metric-points", {"points": [2]})
        return reporter.queues["metric-points"].qsize()

    assert asyncio.run(pending_metrics()) == 2


def test_run_ids_match_other_frameworks():
    async def new_run_id():
        return DojoReporter(CONFIG).run.training_run_id

    # Same create_hash names as Slime/Miles runs: <random words>-<12 hex>.
    run_id = asyncio.run(new_run_id())
    assert not run_id.startswith("spindle-")
    assert re.fullmatch(r".+-[0-9a-f]{12}", run_id)


def test_recipe_lists_every_config_field():
    class Builder:
        __chz_fields__ = {"batch_size": None, "group_size": None}
        batch_size = 4
        group_size = 8

    config = SimpleNamespace(
        **{**vars(CONFIG), "dataset_builder": Builder(), "wandb_api_key": "secret"}
    )

    async def recipe():
        return DojoReporter(config).run.config["recipe"]

    fields = asyncio.run(recipe())
    assert set(vars(CONFIG)) <= set(fields)
    assert fields["dataset_builder"].endswith("Builder")
    assert fields["dataset_builder.group_size"] == 8
    assert fields["wandb_api_key"] == "[redacted]"
