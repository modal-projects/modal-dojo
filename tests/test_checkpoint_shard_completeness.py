import pickle
from types import SimpleNamespace

from modal_dojo.common.torch_dist_checkpoint import (
    is_readable_torch_dist_checkpoint_dir,
)


def test_metadata_does_not_hide_missing_or_truncated_node_shards(tmp_path):
    metadata = SimpleNamespace(
        storage_data={
            0: SimpleNamespace(relative_path="__0.distcp", offset=0, length=4),
            1: SimpleNamespace(relative_path="__1.distcp", offset=2, length=4),
            2: SimpleNamespace(relative_path="__1.distcp", offset=0, length=2),
        }
    )
    (tmp_path / ".metadata").write_bytes(pickle.dumps(metadata))
    (tmp_path / "__0.distcp").write_bytes(b"1234")
    assert not is_readable_torch_dist_checkpoint_dir(tmp_path)
    (tmp_path / "__1.distcp").write_bytes(b"12345")
    assert not is_readable_torch_dist_checkpoint_dir(tmp_path)
    (tmp_path / "__1.distcp").write_bytes(b"123456")
    assert is_readable_torch_dist_checkpoint_dir(tmp_path)


def test_invalid_metadata_is_not_a_cache_hit(tmp_path):
    (tmp_path / ".metadata").write_bytes(b"incomplete")
    (tmp_path / "__0.distcp").write_bytes(b"1234")
    assert not is_readable_torch_dist_checkpoint_dir(tmp_path)
