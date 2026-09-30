"""Exercise the pinned engine-readiness method with stateful memory RPCs."""

import ast
import asyncio
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_lora_initial_offload as patcher,
)

SOURCE = Path(__file__).parent / "testdata/miles/k3_startup/server_cell.py.input"


class MemoryClient:
    def __init__(self):
        self.offloaded = set()
        self.calls = []

    async def release_memory_occupation(self, tags=None):
        tags = set(tags or ("weights", "kv", "graph"))
        assert not self.offloaded & tags
        self.offloaded.update(tags)
        self.calls.append(("release", tags))

    async def resume_memory_occupation(self, tags=None):
        tags = set(tags or ("weights", "kv", "graph"))
        assert tags <= self.offloaded
        self.offloaded.difference_update(tags)
        self.calls.append(("resume", tags))


def patched_method(tmp_path, client, healthy=True):
    path = tmp_path / "server_cell.py"
    path.write_text(SOURCE.read_text())
    patcher.apply(path)
    source = path.read_text()
    assert source == SOURCE.with_suffix(".output").read_text()
    patcher.apply(path)
    assert source == path.read_text()
    compile(source, str(path), "exec")
    tree = ast.parse(source)
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_tick_when_initializing"
    )
    namespace = {
        "os": os,
        "logger": logging.getLogger(__name__),
        "GPU_MEMORY_TYPE_WEIGHTS": "weights",
        "GPU_MEMORY_TYPE_KV_CACHE": "kv",
        "GPU_MEMORY_TYPE_CUDA_GRAPH": "graph",
        "SGLangApiClient": lambda **kw: client,
        "probe_server_healthy": AsyncMock(return_value=healthy),
        "StateInitializing": object(),
        "StatePendingWeights": lambda **kw: SimpleNamespace(**kw),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method.name]


def cell(**overrides):
    args = dict(
        lora_rank=32,
        debug_rollout_only=False,
        check_weight_update_equal=False,
        check_weight_update_skip_list=[],
    )
    args.update(overrides)
    return SimpleNamespace(
        args=SimpleNamespace(**args),
        meta=SimpleNamespace(
            needs_offload=True, update_weights=True, sglang_api_key=None
        ),
        _state=SimpleNamespace(addr_info=SimpleNamespace(server_url="http://engine")),
        check_weights=AsyncMock(),
        _register_with_router=AsyncMock(),
        _mark_serving=Mock(),
        _change_state=Mock(),
    )


@pytest.mark.parametrize("fast", [False, True])
def test_initial_update_and_next_offload_preserve_memory_state(
    tmp_path, monkeypatch, fast
):
    monkeypatch.setenv("MODAL_DOJO_K3_KEEP_INITIAL_BASE_WEIGHTS", str(int(fast)))
    client = MemoryClient()
    instance = cell()
    initialize = patched_method(tmp_path, client)

    async def lifecycle():
        await initialize(instance)
        assert client.offloaded == {"kv", "graph"}
        assert instance._change_state.call_args.args[0] == "mark_pending_weights"
        assert not instance._register_with_router.called
        expected = (
            [("release", {"kv", "graph"})]
            if fast
            else [("release", {"weights", "kv", "graph"}), ("resume", {"weights"})]
        )
        assert client.calls == expected
        # Weight-sync finalize restores KV/graphs, then rollout can execute.
        await client.resume_memory_occupation(tags=["kv", "graph"])
        assert not client.offloaded
        # The next normal training transition must still offload the weights.
        await client.release_memory_occupation(tags=["kv", "graph"])
        await client.release_memory_occupation(tags=["weights"])
        await client.resume_memory_occupation(tags=["weights"])
        await client.resume_memory_occupation(tags=["kv", "graph"])
        assert not client.offloaded

    asyncio.run(lifecycle())


@pytest.mark.parametrize(
    "override",
    [
        {"lora_rank": 0},
        {"lora_rank": None},
        {"debug_rollout_only": True},
        {"check_weight_update_equal": True},
    ],
)
def test_other_modes_keep_upstream_sequence(tmp_path, monkeypatch, override):
    monkeypatch.setenv("MODAL_DOJO_K3_KEEP_INITIAL_BASE_WEIGHTS", "1")
    client = MemoryClient()
    instance = cell(**override)
    asyncio.run(patched_method(tmp_path, client)(instance))
    assert client.calls == [
        ("release", {"weights", "kv", "graph"}),
        ("resume", {"weights"}),
    ]
    if override.get("check_weight_update_equal"):
        assert [c.kwargs["action"] for c in instance.check_weights.call_args_list] == [
            "snapshot",
            "reset_tensors",
        ]


@pytest.mark.parametrize("mode", ["not_healthy", "no_offload", "frozen"])
def test_readiness_guards(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("MODAL_DOJO_K3_KEEP_INITIAL_BASE_WEIGHTS", "1")
    client = MemoryClient()
    instance = cell()
    instance.meta.needs_offload = mode != "no_offload"
    instance.meta.update_weights = mode != "frozen"
    asyncio.run(
        patched_method(tmp_path, client, healthy=mode != "not_healthy")(instance)
    )
    if mode == "frozen":
        assert client.calls == [
            ("release", {"weights", "kv", "graph"}),
            ("resume", {"weights"}),
        ]
        instance._mark_serving.assert_called_once()
    else:
        assert not client.calls
    assert instance._change_state.called == (mode != "not_healthy")


def test_patch_rejects_source_drift_without_writing(tmp_path):
    path = tmp_path / "server_cell.py"
    source = SOURCE.read_text().replace(patcher.ANCHOR, "            pass\n")
    path.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected LoRA initialization source"):
        patcher.apply(path)
    assert path.read_text() == source
