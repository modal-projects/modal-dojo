"""
image: radixark/miles:glm53next
file: megatron/core/transformer/moe/experts.py (image Megatron-LM)

Run grouped experts over row chunks so hot experts at long context do not
materialize every intermediate for all routed tokens at once.
"""

from pathlib import Path

MARKER = "PATCHED_MODAL_DOJO_GLM53_MOE_ROW_CHUNK"

APPENDIX = """

def _modal_dojo_row_chunked(forward, rows=262144):
    import functools

    @functools.wraps(forward)
    def wrapped(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs, *args, **kwargs):
        x = permuted_local_hidden_states
        counts = tokens_per_expert.tolist() if torch.is_tensor(tokens_per_expert) else list(tokens_per_expert)
        if x.shape[0] <= rows or sum(counts) != x.shape[0]:
            return forward(self, x, tokens_per_expert, permuted_probs, *args, **kwargs)
        bounds, start = [], 0
        for count in counts:
            bounds.append((start, start + count))
            start += count
        output, bias = None, None
        for begin in range(0, x.shape[0], rows):
            end = min(begin + rows, x.shape[0])
            splits = [max(0, min(hi, end) - max(lo, begin)) for lo, hi in bounds]
            if torch.is_tensor(tokens_per_expert):
                splits = torch.tensor(splits, dtype=tokens_per_expert.dtype, device=tokens_per_expert.device)
            chunk, bias = forward(self, x[begin:end], splits, permuted_probs[begin:end], *args, **kwargs)
            if output is None:
                output = chunk.new_empty((x.shape[0], *chunk.shape[1:]))
            output[begin:end] = chunk
            del chunk
        return output, bias

    return wrapped


def _modal_dojo_patch_experts():
    import inspect

    expected = ["self", "permuted_local_hidden_states", "tokens_per_expert", "permuted_probs"]
    patched = []
    for name, cls in list(globals().items()):
        forward = inspect.isclass(cls) and cls.__module__ == __name__ and cls.__dict__.get("forward")
        if forward and list(inspect.signature(forward).parameters)[:4] == expected:
            cls.forward = _modal_dojo_row_chunked(forward)
            patched.append(name)
    if not patched:
        raise RuntimeError("modal-dojo MoE row chunk: no grouped experts forward found")


_modal_dojo_patch_experts()
"""


def patch_source(source: str) -> str:
    if MARKER in source:
        return source
    if "import torch\n" not in source:
        raise ValueError("Megatron experts changed: expected `import torch`")
    return f"# {MARKER}\n" + source + APPENDIX


def main() -> None:
    from importlib.util import find_spec

    path = Path(find_spec("megatron.core").origin).parent / "transformer/moe/experts.py"
    path.write_text(patch_source(path.read_text()))
    print(f"Applied {MARKER} to {path}")


if __name__ == "__main__":
    main()
