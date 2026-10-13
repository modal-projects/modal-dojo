"""Fuse K3's SiTU pointwise operations to avoid full-sized FP32 temporaries.

image: radixark/miles:dev-202609251434
commit: 41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles_plugins/models/kimi_k3/ops.py::situ_and_mul

The arithmetic stays upstream-owned. Inductor fuses forward and backward;
dynamic shapes accommodate changing routed-token counts without compiling
one graph per expert load. Used only by the 256k H200 recipe.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles_plugins/models/kimi_k3/ops.py")
MARKER = "PATCHED_K3_SITU_COMPILE"
ANCHOR = "\ndef situ_and_mul(\n"
REPLACEMENT = (
    f"\n# {MARKER}\n@torch.compile(fullgraph=True, dynamic=True)\ndef situ_and_mul(\n"
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(f"{target}: unexpected SiTU compile patch")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one SiTU activation definition")
    source = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"{MARKER}: fused forward/backward with dynamic routed-token shapes")


if __name__ == "__main__":
    apply()
