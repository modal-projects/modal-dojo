"""Multinode conversion must keep every node's additive iteration shards."""

import sys

import pytest

from modal_dojo.frameworks.miles.modal_helpers import convert_hf_to_torch_dist


@pytest.mark.parametrize("publication", ["inline", "helper"])
@pytest.mark.parametrize("skip_release", [False, True])
def test_conversion_preserves_iteration_shards_and_matching_tracker(
    monkeypatch, tmp_path, publication, skip_release
):
    save_dir = tmp_path / "checkpoint"
    iteration = save_dir / "iter_0000001"
    iteration.mkdir(parents=True)
    (iteration / "rank0.distcp").write_text("rank zero shard")
    (iteration / ".metadata").write_text("metadata")
    source = f"""
import shutil
from pathlib import Path
from types import SimpleNamespace

args = SimpleNamespace(save={str(save_dir)!r})
source_dir = Path(args.save) / 'iter_0000001'
target_dir = Path(args.save) / 'release'

def publish_as_release(save_dir):
    source_dir = Path(save_dir) / 'iter_0000001'
    target_dir = Path(save_dir) / 'release'
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.move(source_dir, target_dir)

"""
    # Keep the helper definition out of the source, as in real upstream where
    # it is imported. This proves suppressing only an inline move is inadequate.
    if publication == "helper":
        helper = tmp_path / "publication_helper.py"
        helper.write_text(
            "from pathlib import Path\nimport shutil\n"
            + "def publish_as_release"
            + source.split("def publish_as_release", 1)[1]
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "publication_helper", raising=False)
        source = source.split("def publish_as_release", 1)[0]
        source += "from publication_helper import publish_as_release\npublish_as_release(args.save)\n"
    else:
        source = source.split("def publish_as_release", 1)[0]
        source += "shutil.move(source_dir, target_dir)\n"
    source += """with (Path(args.save) / 'latest_checkpointed_iteration.txt').open('w') as f:
    f.write("release")
"""
    monkeypatch.setattr(
        convert_hf_to_torch_dist, "_load_upstream_source", lambda: source
    )
    monkeypatch.setattr(convert_hf_to_torch_dist, "_start_memory_monitor", lambda: None)
    monkeypatch.delenv("CONVERT_DEQUANT_HF_WEIGHTS", raising=False)
    monkeypatch.delenv("CONVERT_DEQUANT_MXFP4", raising=False)
    if skip_release:
        monkeypatch.setenv("SKIP_RELEASE_RENAME", "1")
    else:
        monkeypatch.delenv("SKIP_RELEASE_RENAME", raising=False)

    convert_hf_to_torch_dist.main()

    tracker = (save_dir / "latest_checkpointed_iteration.txt").read_text()
    expected_dir = iteration if skip_release else save_dir / "release"
    assert tracker == ("1" if skip_release else "release")
    assert (expected_dir / "rank0.distcp").read_text() == "rank zero shard"
    assert (expected_dir / ".metadata").is_file()
    assert not (save_dir / ("release" if skip_release else "iter_0000001")).exists()
