"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/Megatron-LM/commit/f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: megatron/core/pipeline_parallel/schedules.py
"""

from pathlib import Path

TARGET = Path("/root/Megatron-LM/megatron/core/pipeline_parallel/schedules.py")
MARKER = "PATCHED_K3_H200_BACKWARD_CACHE"
ANCHOR = """    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
"""
REPLACEMENT = (
    f"""    # {MARKER}
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 4 * 1024**3:
        torch.cuda.empty_cache()

"""
    + ANCHOR
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(f"{target}: unexpected patched backward cache source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected backward cache source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: reclaim unused cache before backward under pressure")


if __name__ == "__main__":
    apply()
