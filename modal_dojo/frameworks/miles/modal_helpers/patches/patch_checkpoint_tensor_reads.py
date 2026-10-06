"""
image: radixark/miles:dev-202609251434
file: torch/distributed/checkpoint/filesystem.py::FileSystemReader.read_data
"""

from pathlib import Path

TARGET = Path(
    "/opt/sglang/lib/python3.12/site-packages/torch/distributed/checkpoint/filesystem.py"
)
MARKER = "PATCHED_CHECKPOINT_TENSOR_READS"
CHANGES = (
    (
        """        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
""",
        f"""        # {MARKER}
        import time as _dojo_time
        _dojo_started = _dojo_time.monotonic()
        _dojo_reads = _dojo_bytes = 0
        _dojo_read_s = _dojo_copy_s = 0.0
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {{}}
""",
    ),
    (
        """                # TODO sort by offset and cache the reading
                for req in reqs:
""",
        """                # Resharding may request several slices of one stored tensor.
                # Keep only that tensor until its slices have been consumed.
                # Preserve extension behavior and original ordering for transformed files.
                _dojo_cacheable = all(
                    not getattr(self.storage_data[r.storage_index], "transform_descriptors", None)
                    for r in reqs
                )
                if _dojo_cacheable:
                    reqs = sorted(reqs, key=lambda r: (
                        self.storage_data[r.storage_index].offset,
                        self.storage_data[r.storage_index].length,
                    ))
                _dojo_key = _dojo_loaded = None
                for req in reqs:
""",
    ),
    (
        """                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
""",
        """                        _dojo_next_key = (item_md.offset, item_md.length)
                        if not _dojo_cacheable or _dojo_next_key != _dojo_key:
                            _dojo_loaded = None
                            _dojo_tick = _dojo_time.monotonic()
                            _dojo_loaded = cast(
                                Tensor,
                                torch.load(seekable, map_location="cpu", weights_only=True),
                            )
                            _dojo_read_s += _dojo_time.monotonic() - _dojo_tick
                            _dojo_reads += 1
                            _dojo_bytes += item_md.length
                            _dojo_key = _dojo_next_key
                        tensor = narrow_tensor_by_index(
                            _dojo_loaded, req.storage_offsets, req.lengths
                        )
""",
    ),
    (
        """                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
""",
        """                        _dojo_tick = _dojo_time.monotonic()
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)
                        _dojo_copy_s += _dojo_time.monotonic() - _dojo_tick
                _dojo_loaded = None

        print(
            f"DOJO_CKPT_READ rank={os.environ.get('RANK', self.rank)} "
            f"files={len(per_file)} items={len(plan.items)} tensor_reads={_dojo_reads} "
            f"serialized_bytes={_dojo_bytes} read_s={_dojo_read_s:.3f} "
            f"copy_s={_dojo_copy_s:.3f} elapsed_s={_dojo_time.monotonic() - _dojo_started:.3f}",
            flush=True,
        )
        fut: Future = Future()
""",
    ),
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    start = source.index("class FileSystemReader(StorageReader):")
    before, block = source[:start], source[start:]
    if MARKER in block:
        if any(block.count(new) != 1 for _, new in CHANGES):
            raise RuntimeError(f"{target}: unexpected patched checkpoint reader")
        return
    for old, new in CHANGES:
        if block.count(old) != 1:
            raise RuntimeError(f"{target}: checkpoint reader source changed")
        block = block.replace(old, new)
    source = before + block
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()
