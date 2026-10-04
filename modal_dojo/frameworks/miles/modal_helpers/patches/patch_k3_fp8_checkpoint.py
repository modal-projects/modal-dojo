"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
megatron-commit: f148a32b4385b758b66a77c9c3ad1641f1295d4b
file: megatron/core/dist_checkpointing/serialization.py
file: miles_plugins/models/kimi_k3/lora.py
"""

from pathlib import Path

SERIALIZATION = Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/serialization.py"
)
LORA = Path("/root/miles/miles_plugins/models/kimi_k3/lora.py")
LOAD_MARKER = "PATCHED_K3_FP8_LOAD_CPU"
LOAD_ANCHOR = "    force_all_tensors_to_non_fp8(sharded_state_dict)\n"
LOAD_REPLACEMENT = f"""    # {LOAD_MARKER}
    from .dict_utils import nested_values
    from ..fp8_utils import dequantize_fp8_tensor, is_float8tensor

    for value in nested_values(sharded_state_dict):
        if hasattr(value, "data") and is_float8tensor(value.data):
            value.data = dequantize_fp8_tensor(value.data).cpu()
"""
LORA_MARKER = "PATCHED_K3_FROZEN_FP8_INIT_RELEASE"
LORA_ANCHOR = """    for parameter in model.parameters():
        parameter.requires_grad = False
    _enable_full_recompute_input_grads(model)
"""
LORA_REPLACEMENT = f"""    for parameter in model.parameters():
        parameter.requires_grad = False
        # {LORA_MARKER}
        clear_init = getattr(parameter, "clear_high_precision_init_val", None)
        if clear_init is not None:
            clear_init()
    _enable_full_recompute_input_grads(model)
"""


def _patched(path: Path, marker: str, anchor: str, replacement: str) -> str:
    source = path.read_text()
    if marker in source:
        if source.count(replacement) != 1 or anchor in source:
            raise RuntimeError(f"{path}: unexpected patched FP8 checkpoint source")
        return source
    if source.count(anchor) != 1:
        raise RuntimeError(f"{path}: unexpected FP8 checkpoint source")
    result = source.replace(anchor, replacement)
    compile(result, str(path), "exec")
    return result


def apply(serialization: Path = SERIALIZATION, lora: Path = LORA) -> None:

    replacements = [
        (
            serialization,
            _patched(serialization, LOAD_MARKER, LOAD_ANCHOR, LOAD_REPLACEMENT),
        ),
        (lora, _patched(lora, LORA_MARKER, LORA_ANCHOR, LORA_REPLACEMENT)),
    ]
    for path, source in replacements:
        path.write_text(source)
    print("Patched K3 FP8 checkpoint loading: CPU templates and frozen-init release")


if __name__ == "__main__":
    apply()
