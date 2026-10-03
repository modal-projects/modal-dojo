"""Execute the pinned health handler, including cancellation/lease cleanup."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_lora_health as patch,
)

SOURCE = Path(__file__).parent / "testdata/sglang/k3_lora_health.py.input"


def test_patch_is_exact_and_idempotent(tmp_path):
    target = tmp_path / "http_server.py"
    original = SOURCE.read_text()
    target.write_text(original)
    patch.apply(target)
    modified = target.read_text()
    reverted = modified
    for old, new in patch.CHANGES:
        reverted = reverted.replace(new, old)
    assert reverted == original
    patch.apply(target)
    assert target.read_text() == modified


@pytest.mark.parametrize("change", ["missing", "duplicate", "partial"])
def test_unexpected_source_fails_closed(tmp_path, change):
    source = SOURCE.read_text()
    if change == "missing":
        source = source.replace(patch.SUCCESS_CLEANUP, "")
    elif change == "duplicate":
        source += source
    else:
        source += "\n# " + patch.MARKER
    target = tmp_path / "http_server.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source


@pytest.mark.parametrize("respond", [True, False])
@pytest.mark.parametrize(
    "enabled,no_backup,slots,names,expected,status",
    [
        (True, True, 1, ["policy"], "policy", None),
        (True, True, 1, [], None, None),
        (False, True, 1, [], None, None),
        (True, False, 1, ["policy"], None, None),
        (True, True, 2, ["policy"], None, None),
        (True, True, 1, ["a", "b"], None, 503),
    ],
)
def test_health_selects_policy_and_releases_lease(
    tmp_path, respond, enabled, no_backup, slots, names, expected, status
):
    target = tmp_path / "http_server.py"
    target.write_text(SOURCE.read_text())
    patch.apply(target)
    clock = [0]
    requests, released = [], []
    manager = NS(
        gracefully_exit=False,
        server_status="up",
        is_generation=True,
        lora_registry=NS(get_all_adapters=lambda: dict.fromkeys(names)),
        rid_to_state={},
        last_receive_tstamp=-1,
    )

    async def generate(obj, request):
        requests.append(obj)
        manager.rid_to_state[obj.rid] = obj
        if respond:
            manager.last_receive_tstamp = 1
            yield {}
        else:
            await asyncio.Event().wait()

    async def sleep(seconds):
        await asyncio.sleep(0)
        clock[0] += 1

    manager.generate_request = generate
    manager._finalize_lora_lease = lambda state: released.append(state)
    namespace = dict(
        _global_state=NS(tokenizer_manager=manager),
        get_lora=lambda: NS(
            enable_lora=enabled, lora_no_cpu_backup=no_backup, max_loras_per_batch=slots
        ),
        get_disagg=lambda: NS(disaggregation_mode="none"),
        DisaggregationMode=NS(NULL=NS(value="none")),
        GenerateReqInput=NS,
        Response=NS,
        ServerStatus=NS(Starting="starting", Up="up", UnHealthy="unhealthy"),
        envs=NS(
            SGLANG_DIAG_BYPASS_HEALTH_GENERATE=NS(get=lambda: False),
            SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=NS(get=lambda: True),
        ),
        time=NS(
            time=lambda: clock[0], strftime=lambda *a: "time", localtime=lambda x: x
        ),
        uuid=NS(uuid4=lambda: NS(hex="unique")),
        asyncio=NS(create_task=asyncio.create_task, sleep=sleep, gather=asyncio.gather),
        logger=logging.getLogger(__name__),
        HEALTH_CHECK_RID_PREFIX="health",
        HEALTH_CHECK_TIMEOUT=2,
    )
    exec("from __future__ import annotations\n" + target.read_text(), namespace)
    result = asyncio.run(
        namespace["health_generate"](NS(url=NS(path="/health_generate")))
    )
    assert result.status_code == (status or (200 if respond else 503))
    assert manager.rid_to_state == {}
    if status is None:
        assert len(requests) == 1
        assert requests[0].lora_path == expected
        assert released == requests
    else:
        assert requests == released == []
