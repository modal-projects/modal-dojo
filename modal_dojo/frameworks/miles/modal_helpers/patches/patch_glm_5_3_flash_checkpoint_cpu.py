"""Offload full-layer recompute checkpoints in the 256k GLM LoRA recipe.

The outer checkpoints retain ~45 GiB/rank of mHC residual streams at 256k.
Scope saved-tensor offload to their initial forward only: backward recompute
and its nested MoE checkpoints still run normally on the GPU.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_CHECKPOINT_CPU"
OLD = """                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    *rotary_pos_emb,
                    padding_mask,
                )"""
NEW = OLD.replace("tensor_parallel.checkpoint(", "_modal_dojo_cpu_checkpoint(", 1)
APPENDIX = """

def _modal_dojo_cpu_checkpoint(function, distribute_saved_activations, *args):
    # CheckpointFunction.forward executes the layer under no_grad. Thus these
    # hooks pack the outer checkpoint inputs, not a layer's internal activations.
    # The context has exited before backward starts recomputing the layer.
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        return tensor_parallel.checkpoint(function, distribute_saved_activations, *args)
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(OLD) != 1:
        raise ValueError("Megatron full-layer checkpoint handler changed")
    return f"# {MARKER}\n" + source.replace(OLD, NEW, 1) + APPENDIX


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("megatron.core").origin).parent
        / "transformer/transformer_block.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
