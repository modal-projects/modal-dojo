"""Fuse K3's FP32 activation intermediates for the H200 64k recipe.

Pinned source: radixark/miles:dev-202609251434, miles_plugins/models/kimi_k3/ops.py.
Inductor generates the forward and backward kernels from the original formula.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles_plugins/models/kimi_k3/ops.py")
MARKER = "PATCHED_K3_FUSED_ACTIVATION"
ANCHOR = """def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    gate, linear = torch.chunk(x.float(), 2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    linear = linear_beta * torch.tanh(linear / linear_beta)
    return (gate * linear).to(x.dtype)
"""
REPLACEMENT = f"# {MARKER}\n@torch.compile(dynamic=True, fullgraph=True)\n" + ANCHOR


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1 or source.count(ANCHOR) != 1:
            raise RuntimeError(f"{target}: unexpected patched K3 activation source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected K3 activation source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: fuse K3 activation intermediates")


if __name__ == "__main__":
    apply()
