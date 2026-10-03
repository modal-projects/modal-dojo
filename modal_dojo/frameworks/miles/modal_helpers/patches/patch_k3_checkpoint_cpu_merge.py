"""Merge checkpoint shards directly on CPU for the H200 K3 profile.

image: radixark/miles:dev-202609251434
Megatron commit: f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: megatron/core/transformer/utils.py

The frozen trainer weights leave insufficient GPU headroom for checkpoint
merges. Use the existing CPU fallback directly, without repeated failed CUDA
allocations, gc.collect(), and empty_cache(). Do not enable low_memory_resume:
that also changes optimizer-state restoration, including transfers to GPU.
"""

from pathlib import Path

TARGET = Path("/root/Megatron-LM/megatron/core/transformer/utils.py")
MARKER = "PATCHED_K3_CHECKPOINT_CPU_MERGE"
ANCHOR = '''@torch.no_grad()
def cat_with_oom_fallback(sub_state_dict):
    """Merge sharded tensor pieces, falling back to CPU if device-side cat OOMs."""
    # miles --low-memory-resume: merge on CPU proactively to keep peak GPU memory low
    # during distributed-optimizer checkpoint load (avoids the device cat allocation).
    from megatron.training import get_args

    if get_args().low_memory_resume:
        return torch.cat([t.cpu() for t in sub_state_dict])
    try:
        return torch.cat(sub_state_dict)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        logger.warning(
            f"CUDA OutOfMemoryError encountered during tensors merging."
            f" Switching to CPU merge. (Error: {e})"
        )
        merged_sub_state_dict = torch.cat([t.cpu() for t in sub_state_dict])
        gc.collect()
        torch.cuda.empty_cache()
        return merged_sub_state_dict
'''
REPLACEMENT = f'''@torch.no_grad()
def cat_with_oom_fallback(sub_state_dict):
    """Merge checkpoint pieces on CPU, preserving the pinned fallback result."""
    # {MARKER}: H200 has no GPU headroom during base-weight load.
    return torch.cat([t.cpu() for t in sub_state_dict])
'''


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1 or ANCHOR in source:
            raise RuntimeError(f"{target}: unexpected patched checkpoint merge source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected checkpoint merge source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: merge K3 checkpoint shards directly on CPU")


if __name__ == "__main__":
    apply()
