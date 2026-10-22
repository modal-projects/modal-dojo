"""Exercise the real patched allocator lifecycle with CPU CUDA/disk doubles."""

import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches.patch_sglang_frozen_weight_backup import (
    TMS_ARCHIVE_SHA256,
    patch_adapter,
    patch_source,
)

FIXTURES = Path(__file__).parent / "testdata/frozen_backup"


@pytest.fixture(scope="module")
def patched(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tms")
    archive = FIXTURES / "upstream.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == TMS_ARCHIVE_SHA256
    with tarfile.open(archive) as bundle:
        bundle.extractall(tmp, filter="data")
    (root,) = tmp.iterdir()
    patch_source(root)
    return root


@pytest.fixture(scope="module")
def binary(patched, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tms-core")
    for name in ("core.cpp", "core.h", "disk_backend.h"):
        shutil.copy(patched / "csrc" / name, tmp / name)
    shutil.copy(FIXTURES / "stub.h", tmp / "utils.h")
    (tmp / "macro.h").write_text('#include "utils.h"\n')
    (tmp / "api_forwarder.h").write_text('#include "utils.h"\n')
    shutil.copy(FIXTURES / "lifecycle.cpp", tmp / "lifecycle.cpp")
    result = tmp / "lifecycle"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-DUSE_CUDA",
            "-UNDEBUG",
            "core.cpp",
            "lifecycle.cpp",
            "-o",
            str(result),
        ],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
    )
    return result


def test_lifecycle(binary):
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "reused_bytes=16" in result.stdout
    assert "written_bytes=24 reused_bytes=0" in result.stdout


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("empty", "nonempty tag"),
        ("missing", "no allocations"),
        ("cpu", "exclusively disk-backed"),
        ("paused", "active allocations"),
        ("device", "single-device pool"),
    ],
)
def test_seal_rejects_unsafe_pools(binary, mode, reason):
    result = subprocess.run(
        [str(binary), mode], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert reason in result.stderr


def test_source_drift_does_not_partially_patch(patched, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(patched, root)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="anchor changed"):
        patch_source(root)
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = tmp_path / "adapter.py"
    path.write_text("""# PATCHED_MODAL_DOJO_SGLANG_WEIGHTS_DISK_BACKUP
class Adapter:
    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)
""")
    patch_adapter(path)
    original = path.read_text()
    patch_adapter(path)
    assert original == path.read_text()
    events = []
    saver = SimpleNamespace(
        seal_disk_backup=lambda **kw: events.append(("seal", kw)) or 16,
        pause=lambda **kw: events.append(("pause", kw)),
    )
    ns = {"_memory_saver": saver, "logger": SimpleNamespace(info=lambda *args: None)}
    exec(compile(original, str(path), "exec"), ns)  # noqa: S102 - execute the patched adapter fixture
    context = SimpleNamespace(
        get_lora=lambda: SimpleNamespace(enable_lora=True),
        get_model=lambda: SimpleNamespace(weight_cache_mode="off"),
    )
    monkeypatch.setitem(
        __import__("sys").modules, "sglang.srt.runtime_context", context
    )
    monkeypatch.delenv("DOJO_REUSE_FROZEN_BASE_BACKUP", raising=False)
    monkeypatch.delenv("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR", raising=False)
    return ns["Adapter"](), events, context


def test_default_unchanged(adapter):
    instance, events, _ = adapter
    instance.pause("weights")
    assert events == [("pause", {"tag": "weights"})]


def test_only_explicit_weight_pause_seals(adapter, monkeypatch):
    instance, events, _ = adapter
    monkeypatch.setenv("DOJO_REUSE_FROZEN_BASE_BACKUP", "1")
    monkeypatch.setenv("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR", "/disk")
    instance.pause("kv_cache")
    instance.pause("weights")
    assert events == [
        ("pause", {"tag": "kv_cache"}),
        ("seal", {"tag": "weights"}),
        ("pause", {"tag": "weights"}),
    ]


@pytest.mark.parametrize("invalid", ["full_weights", "ipc", "no_disk"])
def test_opt_in_rejects_unsupported_serving(adapter, monkeypatch, invalid):
    instance, events, context = adapter
    monkeypatch.setenv("DOJO_REUSE_FROZEN_BASE_BACKUP", "1")
    if invalid != "no_disk":
        monkeypatch.setenv("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR", "/disk")
    if invalid == "full_weights":
        context.get_lora = lambda: SimpleNamespace(enable_lora=False)
    if invalid == "ipc":
        context.get_model = lambda: SimpleNamespace(weight_cache_mode="ipc")
    with pytest.raises(RuntimeError):
        instance.pause("weights")
    assert not events
