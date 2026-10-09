"""
image: slimerl/slime:nightly-dev-20260722a
commit: https://github.com/THUDM/slime/commit/55828f317171c804b924e71f29a851f167c61377
file: slime/slime/backends/megatron_utils/megatron_to_hf/__init__.py::_convert_to_hf_core
"""

from __future__ import annotations

from pathlib import Path

MARKER = "PATCHED_QWEN36_HF_DISPATCH"
ALIGN_MARKER = "PATCHED_QWEN36_EXPERT_ALIGN"
ANCHOR = "def _convert_to_hf_core(args, model_name, name, param):"
REPLACEMENT = f"""def _convert_to_hf_core(args, model_name, name, param):
    # {MARKER}
    _qwen_family = model_name.lower().replace("_", "").replace("-", "").replace(".", "")
    if "qwen36" in _qwen_family or "qwen35" in _qwen_family:
        return convert_qwen3_5_to_hf(args, name, param)
"""
ALIGN_TAIL = f"""

# {ALIGN_MARKER}
def _align_qwen35_expert(hf_name, tensor, args):
    import itertools

    if tensor.ndim != 3 or ".experts." not in hf_name:
        return tensor
    hidden = args.hidden_size
    ffn = getattr(args, "moe_ffn_hidden_size", None) or getattr(
        args, "ffn_hidden_size", 0
    )
    if hf_name.endswith("experts.gate_up_proj"):
        target = (tensor.shape[0], 2 * ffn, hidden)
    elif hf_name.endswith("experts.down_proj"):
        target = (tensor.shape[0], hidden, ffn)
    else:
        return tensor
    if tuple(tensor.shape) == target:
        return tensor
    if tensor.numel() != target[0] * target[1] * target[2]:
        print(
            f"WEIGHT_SYNC_ALIGN skip {{hf_name}} {{tuple(tensor.shape)}} != {{target}}",
            flush=True,
        )
        return tensor
    dims = list(tensor.shape)
    for perm in itertools.permutations(range(3)):
        if tuple(dims[i] for i in perm) == target:
            print(
                f"WEIGHT_SYNC_ALIGN {{hf_name}} {{tuple(tensor.shape)}} -> {{target}} perm={{perm}}",
                flush=True,
            )
            return tensor.permute(*perm).contiguous()
    return tensor


_qwen36_convert_to_hf = convert_to_hf


def convert_to_hf(args, model_name, name, param, *a, **k):
    if ".experts." in name or 64 in tuple(param.shape):
        print(
            f"WEIGHT_SYNC_IN model={{model_name!r}} {{name}} {{tuple(param.shape)}}",
            flush=True,
        )
    out = _qwen36_convert_to_hf(args, model_name, name, param, *a, **k)
    aligned = []
    for hf_name, tensor in out:
        tensor = _align_qwen35_expert(hf_name, tensor, args)
        if ".experts." in hf_name or 64 in tuple(tensor.shape):
            print(f"WEIGHT_SYNC_OUT {{hf_name}} {{tuple(tensor.shape)}}", flush=True)
        aligned.append((hf_name, tensor))
    return aligned
"""


def compact_model_name(model_name: str) -> str:
    return model_name.lower().replace("_", "").replace("-", "").replace(".", "")


def is_qwen35_family(model_name: str) -> bool:
    compact = compact_model_name(model_name)
    return "qwen36" in compact or "qwen35" in compact


def _patch_file(path: Path) -> None:
    source = path.read_text()
    if MARKER not in source:
        if ANCHOR not in source:
            raise RuntimeError(
                f"Could not find convert_to_hf dispatch anchor in {path}"
            )
        source = source.replace(ANCHOR, REPLACEMENT, 1)
        print(f"Patched {path} to route Qwen3.5/3.6 to convert_qwen3_5_to_hf")
    if ALIGN_MARKER not in source:
        source = source + ALIGN_TAIL
        print(f"Patched {path} to align Qwen3.5 fused expert layouts")
    path.write_text(source)


def main() -> None:
    _patch_file(
        Path("/root/slime/slime/backends/megatron_utils/megatron_to_hf/__init__.py")
    )


if __name__ == "__main__":
    main()
