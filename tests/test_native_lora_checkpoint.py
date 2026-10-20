"""Exercise the pinned native save on a follower's initially empty mount."""

import ast
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.common import torch_dist_checkpoint
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_checkpoint_local_dirs as local_dirs,
    patch_native_lora_checkpoint_commit as commit,
)

DATA = Path(__file__).parent / "testdata/miles"


def _function(source, name, namespace):
    tree = ast.parse(source)
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return namespace[name]


@pytest.mark.parametrize("commit_fails", [False, True])
def test_native_follower_save_creates_directory_and_commits(
    tmp_path, monkeypatch, commit_fails
):
    sources = []
    for fixture, patch in (
        ("checkpoint_io_65e785f.py.input", local_dirs),
        ("lora_utils_65e785f.py.input", commit),
    ):
        target = tmp_path / fixture
        target.write_text((DATA / fixture).read_text())
        patch.apply(target)
        patched = target.read_text()
        patch.apply(target)
        assert target.read_text() == patched
        compile(patched, str(target), "exec")
        sources.append(patched)

    # Rank zero prepared its own mount; it cannot create the follower's path.
    dist = SimpleNamespace(
        is_initialized=lambda: True,
        get_rank=lambda: 1,
        get_world_size=lambda: 2,
        broadcast_object_list=lambda *a, **kw: None,
        all_gather_object=lambda *a, **kw: None,
    )
    write_dir = _function(
        sources[0],
        "write_checkpoint_dir",
        dict(
            Path=Path,
            dist=dist,
            get_gloo_group=lambda: None,
            shutil=shutil,
            json=json,
            logger=logging.getLogger(__name__),
        ),
    )
    saved = tmp_path / "follower/iter_0000005/adapter"
    commits = []

    def commit_volume():
        assert (saved / "adapter_megatron_rank1.pt").read_text() == "adapter"
        commits.append(True)
        if commit_fails:
            raise RuntimeError("Volume commit failed")

    monkeypatch.setattr(
        torch_dist_checkpoint, "commit_latest_checkpoint_volume", commit_volume
    )
    save = _function(
        sources[1],
        "save_lora_checkpoint",
        dict(
            Path=Path,
            dist=dist,
            _adapter_shard_params=lambda model: {},
            torch=SimpleNamespace(save=lambda state, path: path.write_text("adapter")),
            write_checkpoint_dir=write_dir,
        ),
    )
    args = SimpleNamespace(megatron_to_hf_mode="raw")
    if commit_fails:
        with pytest.raises(RuntimeError, match="Volume commit failed"):
            save([], args, str(saved), publisher=None)
    else:
        assert save([], args, str(saved), publisher=None) == str(saved)
    assert commits == [True]


def test_native_commit_patch_rejects_changed_upstream(tmp_path):
    target = tmp_path / "utils.py"
    target.write_text("def save_lora_checkpoint():\n    new_save_api()\n")
    with pytest.raises(RuntimeError, match="expected one native LoRA"):
        commit.apply(target)
