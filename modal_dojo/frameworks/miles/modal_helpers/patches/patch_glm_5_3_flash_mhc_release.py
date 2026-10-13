"""
image: radixark/miles:glm53next
file: megatron/core/transformer/transformer_layer.py (image Megatron-LM)

Release the mHC MLP norm manager after its last use, matching TransformerLayer,
so it does not retain restored checkpoint inputs and their gradients.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_MHC_RELEASE"
OLD = """        hidden_states = self.mlp_norm_manager.group_offload(hidden_states)

        output = make_viewless_tensor(
"""
NEW = """        hidden_states = self.mlp_norm_manager.group_offload(hidden_states)
        self.mlp_norm_manager = None

        output = make_viewless_tensor(
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if source.count(OLD) != 1:
        raise ValueError("Megatron mHC MLP manager release site changed")
    return f"# {MARKER}\n" + source.replace(OLD, NEW, 1)


def main() -> None:
    from importlib.util import find_spec

    path = (
        Path(find_spec("megatron.core").origin).parent
        / "transformer/transformer_layer.py"
    )
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
