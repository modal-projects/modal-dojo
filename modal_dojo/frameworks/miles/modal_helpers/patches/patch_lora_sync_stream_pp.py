"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator.py::MegatronHfWeightIteratorBase._iter_hf_adapter_units
"""

import pathlib

MARKER = "PATCHED_LORA_SYNC_STREAM_PP_BOUNDED"

TARGET = pathlib.Path(
    "/root/miles/miles/backends/megatron_utils/update_weight/hf_weight_iterator.py"
)
ANCHOR = "def _gather_pp_full_adapter("

OVERRIDE = f"""

# {MARKER}
def _iter_hf_adapter_units_streaming(self, adapter, *, materialize):
    named_tensors = self._export_pp_local_lora(adapter)
    pp = get_parallel_state().pp
    if not self.placement.gather_pp or pp.size == 1:
        if not materialize:
            return
        _check_adapter_export([n for n, _ in named_tensors])
        while named_tensors:
            hf_name, tensor = named_tensors.pop(0)
            yield [(hf_name, tensor)]
        return

    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()
    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in named_tensors]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)
    if materialize:
        _check_adapter_export([n for meta in all_meta for n, _, _ in meta])

    local_by_name = {{n: t for n, t in named_tensors}}
    del named_tensors
    max_bytes = self.args.update_weight_buffer_size
    for src, meta in enumerate(all_meta):
        for dtype, entries, numel in _lora_pp_chunks(meta, max_bytes):
            # Fresh storage: the downstream packer can retain views while
            # requesting the next chunk. Never overwrite a reusable buffer.
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src == pp.rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name.pop(n).reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src], group=pp.group)
            if materialize:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    yield [(n, flat[off : off + k].view(shape))]
                    off += k
            del flat


def _lora_pp_chunks(meta, max_bytes):
    if max_bytes <= 0:
        raise ValueError("LoRA PP broadcast buffer size must be positive")
    by_dtype: dict = {{}}
    for name, shape, dtype in meta:
        by_dtype.setdefault(dtype, []).append((name, shape))
    for dtype, entries in by_dtype.items():
        chunk = []
        numel = 0
        for name, shape in entries:
            size = math.prod(shape)
            if chunk and (numel + size) * dtype.itemsize > max_bytes:
                yield dtype, chunk, numel
                chunk, numel = [], 0
            chunk.append((name, shape))
            numel += size
            # Match Miles' atomic-tensor contract: an individual tensor may
            # exceed the target, but must never share that oversized chunk.
            if numel * dtype.itemsize >= max_bytes:
                yield dtype, chunk, numel
                chunk, numel = [], 0
        if chunk:
            yield dtype, chunk, numel


def _check_adapter_export(names):
    if not names:
        raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
    if not any(is_lora_weight_name(name) for name in names):
        raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B tensors")


MegatronHfWeightIteratorBase._iter_hf_adapter_units = _iter_hf_adapter_units_streaming
"""


def apply(target: pathlib.Path = TARGET) -> None:
    src = target.read_text()
    if MARKER in src:
        if src.count(OVERRIDE) != 1:
            raise RuntimeError("Unexpected partially patched LoRA PP iterator")
        return
    if "PATCHED_LORA_SYNC_STREAM_PP" in src:
        raise RuntimeError(
            "Rebuild from the pinned image before replacing the old PP patch"
        )
    if (
        src.count(ANCHOR) != 1
        or src.count("def _iter_hf_adapter_units(self, adapter, *, materialize):") != 1
    ):
        raise RuntimeError(
            "LoRA sync streaming patch did not find the expected iterator; "
            "inspect the new Miles source before applying."
        )
    patched = src.rstrip("\n") + "\n" + OVERRIDE
    compile(patched, str(target), "exec")
    target.write_text(patched)
    print(
        "Patched MegatronHfWeightIteratorBase._iter_hf_adapter_units to bound PP broadcasts"
    )


if __name__ == "__main__":
    apply()
