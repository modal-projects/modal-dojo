"""Accept canonical HF cache shard paths in the pinned Muse loader.

image: radixark/miles:muse-spark-open (digest pinned in muse_spark.py)
commit: https://github.com/radixark/miles/commit/002ec9cf35e20dab56bddc8b55e4364c407a2231
file: miles_plugins/models/muse_spark/checkpoint.py::SafetensorsCheckpoint

The upstream containment check still rejects snapshot -> blobs symlinks.
TODO: remove when upstream accepts canonical Hugging Face cache shards while
retaining its path-traversal checks.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles_plugins/models/muse_spark/checkpoint.py")


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    marker = "def _dojo_resolve_safetensors_shard(checkpoint_dir, shard_filename):"
    resolved_reader = 'with safe_open(self._shard_paths[self.weight_map[name]], framework="pt", device="cpu") as shard_reader:'
    helper = '''def _dojo_resolve_safetensors_shard(checkpoint_dir, shard_filename):
    """Accept local shards and canonical HF cache blobs without allowing traversal."""
    import os
    import re

    root = Path(checkpoint_dir).resolve()
    relative = Path(shard_filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid safetensors shard path: {shard_filename}")
    candidate = root / relative
    shard = candidate.resolve()
    allowed = shard.is_relative_to(root)
    if not allowed and root.parent.name == "snapshots":
        repository = root.parent.parent
        blobs = repository / "blobs"
        allowed = (
            repository.name.startswith("models--")
            and blobs.is_dir()
            and blobs.resolve() == blobs
            and shard.parent == blobs
        )
        # New HF Hub caches deduplicate each repository blob through a second
        # symlink into hub/blobs/<2 hex>/<64 hex>. Require the first hop to be
        # this repository's blob, not a direct global-store or sibling-repo link.
        if not allowed and candidate.is_symlink() and candidate.parent.resolve().is_relative_to(root):
            first_hop = Path(os.path.normpath(str(candidate.parent / os.readlink(candidate))))
            global_blobs = repository.parent / "blobs"
            allowed = (
                repository.name.startswith("models--")
                and blobs.is_dir()
                and blobs.resolve() == blobs
                and first_hop.parent == blobs
                and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", first_hop.name) is not None
                and first_hop.is_symlink()
                and first_hop.resolve() == shard
                and global_blobs.is_dir()
                and global_blobs.resolve() == global_blobs
                and shard.parent.parent == global_blobs
                and re.fullmatch(r"[0-9a-f]{64}", shard.name) is not None
                and shard.parent.name == shard.name[:2]
            )
    if not allowed or not shard.is_file():
        raise ValueError(f"invalid safetensors shard path: {shard_filename}")
    return shard


'''
    if marker in source:
        assert resolved_reader in source, "Incomplete Muse HF cache shard patch"
        assert "self._shard_paths[shard_filename] = shard_path" in source
        start = source.index(marker)
        end = source.index("class SafetensorsCheckpoint:\n", start)
        source = source[:start] + helper + source[end:]
        compile(source, str(target), "exec")
        target.write_text(source)
        print("Muse HF cache shard patch current:", target)
        return
    old = """            shard_path = (self.path / shard_filename).resolve()
            if not shard_path.is_relative_to(self.path) or not shard_path.is_file():
                raise ValueError(f"invalid safetensors shard path: {shard_filename}")
"""
    new = """            shard_path = _dojo_resolve_safetensors_shard(self.path, shard_filename)
            self._shard_paths[shard_filename] = shard_path
"""
    replacements = [
        ("class SafetensorsCheckpoint:\n", helper + "class SafetensorsCheckpoint:\n"),
        (
            "        self.tensor_metadata = {}\n",
            "        self.tensor_metadata = {}\n        self._shard_paths = {}\n",
        ),
        (old, new),
        (
            'with safe_open(self.path / self.weight_map[name], framework="pt", device="cpu") as shard_reader:',
            resolved_reader,
        ),
    ]
    for old_text, new_text in replacements:
        assert source.count(old_text) == 1, (
            f"Muse checkpoint source changed: {old_text[:80]!r}"
        )
        source = source.replace(old_text, new_text)
    compile(source, str(target), "exec")
    target.write_text(source)
    print("Muse HF cache shard patch applied:", target)


if __name__ == "__main__":
    apply()
