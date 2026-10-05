"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/Megatron-LM/commit/f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: megatron/core/transformer/transformer_block.py
"""

from pathlib import Path

TARGET = Path("/root/Megatron-LM/megatron/core/transformer/transformer_block.py")
MARKER = "PATCHED_K3_H200_CHECKPOINT_OFFLOAD"
ANCHOR = """                hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))
"""
REPLACEMENT = f"""                # {MARKER}
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, chunk_end))
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(REPLACEMENT) != 1:
            raise RuntimeError(
                f"{target}: unexpected patched checkpoint offload source"
            )
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected checkpoint offload source")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: offload uniform recompute checkpoint tensors to CPU")


if __name__ == "__main__":
    apply()
