"""Pinned-source contract and tensor lifetime at the engine commit boundary."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_ipc_bucket_empty_cache as patcher,
    patch_lora_sync_stream_pp as stream_patcher,
)

FIXTURES = Path(__file__).parent / "testdata/miles/kimi_k3"


def test_streaming_export_snapshot(tmp_path):
    target = tmp_path / "hf_weight_iterator.py"
    target.write_text((FIXTURES / "hf_weight_iterator.py.input").read_text())
    stream_patcher.apply(target)
    expected = (FIXTURES / "hf_weight_iterator.py.output").read_text()
    assert target.read_text() == expected
    compile(expected, str(target), "exec")
    stream_patcher.apply(target)
    assert target.read_text() == expected


def test_streaming_export_rejects_source_drift(tmp_path):
    target = tmp_path / "hf_weight_iterator.py"
    target.write_text("# changed upstream\n")
    with pytest.raises(RuntimeError, match="expected iterator"):
        stream_patcher.apply(target)
    assert target.read_text() == "# changed upstream\n"


@pytest.fixture
def patched(tmp_path):
    (tmp_path / "protocols").mkdir()
    for name, target in (
        ("cuda_ipc.py", tmp_path / "protocols/cuda_ipc.py"),
        ("updater.py", tmp_path / "updater.py"),
    ):
        target.write_text((FIXTURES / f"{name}.input").read_text())
    patcher.apply(tmp_path)
    return tmp_path


def test_snapshots_and_idempotence(patched):
    for name, target in (
        ("cuda_ipc.py", patched / "protocols/cuda_ipc.py"),
        ("updater.py", patched / "updater.py"),
    ):
        actual = target.read_text()
        assert actual == (FIXTURES / f"{name}.output").read_text()
        compile(actual, str(target), "exec")
        patcher.apply(patched)
        assert target.read_text() == actual


def test_drift_fails_before_writing_either_file(tmp_path):
    (tmp_path / "protocols").mkdir()
    updater = tmp_path / "updater.py"
    original = (FIXTURES / "updater.py.input").read_text()
    updater.write_text(original)
    (tmp_path / "protocols/cuda_ipc.py").write_text("# upstream changed\n")
    with pytest.raises(RuntimeError, match="expected one patch anchor"):
        patcher.apply(tmp_path)
    assert updater.read_text() == original


@pytest.mark.parametrize("empty", [False, True])
def test_final_bucket_released_before_engine_commit(patched, empty):
    tree = ast.parse((patched / "updater.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "update_weights"
    )
    method.decorator_list = []
    events = []
    refs = []

    class Tensor:
        pass

    def weights(*args, **kwargs):
        if not empty:
            tensor = Tensor()
            refs.append(weakref.ref(tensor))
            yield [("adapter", tensor)]

    def finalize(version):
        assert all(ref() is None for ref in refs)
        assert events[-1] == "barrier"
        events.append("finalize")

    protocol = SimpleNamespace(
        begin_sync=lambda *a: True,
        needs_base_resync_for_lora=False,
        use_weight_update_session=True,
        rollout_engines=[],
        is_sender=True,
        group_name="test",
        send_bucket=lambda b: events.append("send"),
        after_base_weights=lambda: events.append("sent"),
        finalize=finalize,
        after_engines_resumed=lambda: events.append("resumed"),
    )
    obj = SimpleNamespace(
        protocol=protocol,
        weight_version=0,
        _iter_base_buckets=None,
        is_lora=True,
        _get_updated_adapters=lambda: [],
        _register_new_lora_adapters=lambda *a: None,
        args=SimpleNamespace(check_lora_weight_equal=False),
        _hf_weight_iterator=SimpleNamespace(
            weight_update_selector="all", iter_hf_weights=weights
        ),
        weights_getter=lambda: None,
    )
    ns = {
        "dist": SimpleNamespace(
            get_rank=lambda: 0, barrier=lambda **k: events.append("barrier")
        ),
        "get_gloo_group": lambda: None,
        "timer": lambda *a: nullcontext(),
        "tqdm": Mock(),
        "pause_engines": Mock(),
        "begin_weight_update": Mock(),
        "end_weight_update": lambda *a, **k: events.append("commit"),
        "set_weight_version": Mock(),
        "resume_engines": Mock(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "updater.py", "exec"), ns)
    ns["update_weights"](obj)
    assert events.index("finalize") < events.index("commit")
