"""Checkpoint writer behavior with separate per-container filesystem views."""

import ast
import json
import logging
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_checkpoint_local_dirs as patcher,
)

FIXTURES = Path(__file__).parent / "testdata/miles/kimi_k3"


def _writer(source, rank):
    method = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "write_checkpoint_dir"
    )

    def gather(output, value, **kwargs):
        output[:] = [None, value]

    ns = {
        "Path": Path,
        "Callable": __import__("collections.abc", fromlist=["Callable"]).Callable,
        "json": json,
        "shutil": shutil,
        "logger": logging.getLogger(__name__),
        "get_gloo_group": lambda: None,
        "dist": SimpleNamespace(
            is_initialized=lambda: True,
            get_rank=lambda: rank,
            get_world_size=lambda: 2,
            broadcast_object_list=lambda *a, **k: None,
            all_gather_object=gather,
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "checkpoint_io.py", "exec"),
        ns,
    )
    return ns["write_checkpoint_dir"]


def test_missing_directory_on_nonzero_rank(tmp_path):
    source = (FIXTURES / "checkpoint_io.py.input").read_text()
    path = tmp_path / "node1" / "iter_0000000" / "adapter"

    def write(directory):
        (directory / "rank1.pt").write_bytes(b"adapter")

    with pytest.raises(FileNotFoundError):
        _writer(source, rank=1)(path, write)
    target = tmp_path / "checkpoint_io.py"
    target.write_text(source)
    patcher.apply(target)
    _writer(target.read_text(), rank=1)(path, write)
    assert (path / "rank1.pt").read_bytes() == b"adapter"
    assert not (path / "META.json").exists()


def test_snapshot_and_idempotence(tmp_path):
    target = tmp_path / "checkpoint_io.py"
    target.write_text((FIXTURES / "checkpoint_io.py.input").read_text())
    patcher.apply(target)
    expected = (FIXTURES / "checkpoint_io.py.output").read_text()
    assert target.read_text() == expected
    patcher.apply(target)
    assert target.read_text() == expected
    compile(expected, str(target), "exec")


def test_drift_rejected(tmp_path):
    target = tmp_path / "checkpoint_io.py"
    target.write_text("# changed upstream\n")
    with pytest.raises(RuntimeError, match="expected one checkpoint write anchor"):
        patcher.apply(target)
    assert target.read_text() == "# changed upstream\n"


def test_local_directory_error_joins_error_collection(tmp_path):
    path = tmp_path / "not_a_directory"
    path.write_text("blocked")
    source = (FIXTURES / "checkpoint_io.py.output").read_text()
    with pytest.raises(FileExistsError):
        _writer(source, rank=1)(path, lambda _: pytest.fail("writer must not run"))


def test_rank_zero_overwrite_and_completion(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "old.pt").write_bytes(b"old")
    write = _writer((FIXTURES / "checkpoint_io.py.output").read_text(), rank=0)
    with pytest.raises(FileExistsError):
        write(path, lambda _: None, overwrite=False)
    write(
        path,
        lambda p: (p / "new.pt").write_bytes(b"new"),
        metadata={"step": 0},
        completion_marker="DONE",
    )
    assert not (path / "old.pt").exists()
    assert json.loads((path / "META.json").read_text()) == {"step": 0}
    assert (path / "DONE").exists()
