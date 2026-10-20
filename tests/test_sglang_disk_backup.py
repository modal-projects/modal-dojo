"""Exercise the actual pinned adapter hook and its opt-in boundary."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_sglang_weights_disk_backup as patch,
)


@pytest.mark.parametrize(
    "enabled,tag,backup,uses_disk",
    [
        (True, "weights", True, True),
        (False, "weights", True, False),
        (True, "weights", False, False),
        (True, "kv_cache", True, False),
        (True, "cuda_graph", True, False),
    ],
)
def test_pinned_adapter_disk_backup_is_opt_in(
    tmp_path, monkeypatch, enabled, tag, backup, uses_disk
):
    target = tmp_path / "adapter.py"
    fixture = Path(__file__).parent / "testdata/sglang/memory_saver_34033fe.py.input"
    target.write_text(fixture.read_text())
    patch.apply(target)
    patched = target.read_text()
    patch.apply(target)
    assert target.read_text() == patched
    tree = ast.parse(patched)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_TorchMemorySaverAdapterReal"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "region"
    )
    disk_dirs = []
    namespace = {
        "_memory_saver": SimpleNamespace(
            region=lambda **kw: kw, set_disk_backup_dir=disk_dirs.append
        )
    }
    exec(ast.unparse(method), namespace)
    root = tmp_path / "offload"
    if enabled:
        monkeypatch.setenv("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR", str(root))
    else:
        monkeypatch.delenv("DOJO_SGLANG_WEIGHTS_DISK_BACKUP_DIR", raising=False)
    result = namespace["region"](None, tag, backup)
    assert result == (
        dict(tag=tag, enable_cpu_backup=False, enable_disk_backup=True)
        if uses_disk
        else dict(tag=tag, enable_cpu_backup=backup)
    )
    assert disk_dirs == ([str(root / str(os.getpid()))] if uses_disk else [])
    assert root.exists() == uses_disk


def test_unknown_sglang_hook_is_rejected(tmp_path):
    target = tmp_path / "adapter.py"
    target.write_text("new_memory_api()\n")
    with pytest.raises(RuntimeError, match="expected pinned SGLang"):
        patch.apply(target)
