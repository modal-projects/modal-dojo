import pickle
from types import SimpleNamespace

import pytest

from modal_dojo.common import torch_dist_checkpoint as checkpoint


def write_metadata(root, entries):
    (root / ".metadata").write_bytes(
        pickle.dumps(
            SimpleNamespace(
                storage_data={
                    index: SimpleNamespace(
                        relative_path=path, offset=offset, length=length
                    )
                    for index, (path, offset, length) in enumerate(entries)
                }
            )
        )
    )


def test_metadata_requires_all_referenced_shards_and_full_ranges(tmp_path):
    # The old presence check accepts this checkpoint even with a whole node absent.
    (tmp_path / "__0_0.distcp").write_bytes(b"0123456789")
    write_metadata(tmp_path, [("__0_0.distcp", 0, 10), ("__40_0.distcp", 3, 7)])
    assert checkpoint.is_complete_torch_dist_checkpoint_dir(tmp_path)
    assert not checkpoint.is_complete_torch_dist_checkpoint_storage(tmp_path)
    (tmp_path / "__40_0.distcp").write_bytes(b"012345678")
    assert not checkpoint.is_complete_torch_dist_checkpoint_storage(tmp_path)
    (tmp_path / "__40_0.distcp").write_bytes(b"0123456789")
    assert checkpoint.is_complete_torch_dist_checkpoint_storage(tmp_path)


@pytest.mark.parametrize(
    "entry",
    [
        ("../outside.distcp", 0, 1),
        ("/outside.distcp", 0, 1),
        ("__0_0.distcp", -1, 1),
        ("__0_0.distcp", 0, -1),
        ("__0_0.distcp", "0", 1),
        (".metadata", 0, 1),
    ],
)
def test_invalid_storage_range_is_incomplete(tmp_path, entry):
    (tmp_path / "__0_0.distcp").write_bytes(b"x")
    write_metadata(tmp_path, [entry])
    assert not checkpoint.is_complete_torch_dist_checkpoint_storage(tmp_path)


def test_empty_or_corrupt_metadata_is_incomplete(tmp_path):
    (tmp_path / "__0_0.distcp").write_bytes(b"x")
    for payload in [b"", b"corrupt", pickle.dumps(SimpleNamespace(storage_data={}))]:
        (tmp_path / ".metadata").write_bytes(payload)
        assert not checkpoint.is_complete_torch_dist_checkpoint_storage(tmp_path)


@pytest.mark.parametrize("coordinator", [False, True])
def test_volume_commit_is_synchronized_before_worker_exit(monkeypatch, coordinator):
    events = []
    monkeypatch.setattr(checkpoint, "_is_node_leader", lambda: True)
    monkeypatch.setattr(
        checkpoint, "_commit_volume", lambda name: events.append("commit")
    )

    def gather(error):
        events.append("gather")
        return [error, None]

    dist = SimpleNamespace(is_coordinator=coordinator, all_gather_object=gather)
    checkpoint.commit_checkpoint_volume_across_ranks(
        "shared", dist, lambda: events.append("barrier")
    )
    assert events == (
        ["gather", "barrier", "commit", "gather", "barrier"]
        if coordinator
        else ["commit", "gather", "barrier", "gather", "barrier"]
    )


def test_worker_commit_failure_is_propagated_to_coordinator(monkeypatch):
    monkeypatch.setattr(checkpoint, "_is_node_leader", lambda: True)
    dist = SimpleNamespace(
        is_coordinator=True, all_gather_object=lambda _: [None, "disk error"]
    )
    with pytest.raises(RuntimeError, match="shard commit failed.*disk error"):
        checkpoint.commit_checkpoint_volume_across_ranks("shared", dist, lambda: None)
