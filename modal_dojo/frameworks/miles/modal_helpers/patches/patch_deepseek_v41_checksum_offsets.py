"""Use 64-bit offsets when SGLang hashes DeepSeek's large weight tensors.

The image's post-sync checksum audit hashes every model tensor. Its
Triton hash kernel multiplies a signed 32-bit program ID by the tile size,
overflowing beyond 2**31 uint32 words (8 GiB) and reading invalid addresses.
Promote before multiplication; the intentional uint32 hash mixing stays intact.

Image: radixark/miles:dev-202610082133
SGLang: b84bec5761a6cdc8b1df86b9b89ead80a7a9bb02
"""

from pathlib import Path

TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/kernels/ops/memory/gpu_tensor_hash.py"
)
MARKER = "PATCHED_DEEPSEEK_V41_CHECKSUM_OFFSETS"
OLD = "    base = pid * TILE\n"
NEW = f"    base = pid.to(tl.int64) * TILE  # {MARKER}\n"


if __name__ == "__main__":
    source = TARGET.read_text()
    if MARKER not in source:
        if source.count(OLD) != 1:
            raise ValueError("DeepSeek checksum offset patch did not match")
        TARGET.write_text(source.replace(OLD, NEW, 1))
    print("Patched large-tensor checksum offsets")
