"""Reuse expert LoRA delta storage without mutating TE's autograd output view.

image: radixark/miles:dev-202609251434
commit: 41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles_plugins/models/kimi_k3/lora.py::_apply_expert_lora
"""

from pathlib import Path

TARGET = Path("/root/miles/miles_plugins/models/kimi_k3/lora.py")
MARKER = "PATCHED_K3_LORA_DELTA_REUSE"
HELPER = '''# PATCHED_K3_LORA_DELTA_REUSE
class _AddIntoLoRADelta(torch.autograd.Function):
    """The fresh LoRA delta is disposable; TE's output view is not."""

    @staticmethod
    def forward(ctx, output, delta, scale):
        ctx.scale = scale
        ctx.mark_dirty(delta)
        return torch.add(output, delta, alpha=scale, out=delta)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * ctx.scale, None


'''


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if source.count(HELPER) != 1 or source.count("_AddIntoLoRADelta.apply(") != 2:
            raise RuntimeError(f"{target}: unexpected LoRA delta reuse patch")
        return
    start_anchor = "def _apply_expert_lora(\n"
    end_anchor = "\ndef apply_kimi_k3_lora("
    if source.count(start_anchor) != 1 or source.count(end_anchor) != 1:
        raise RuntimeError(f"{target}: missing expert LoRA boundaries")
    start = source.index(start_anchor)
    end = source.index(end_anchor, start)
    expert = source[start:end]
    old = "torch.add(output, delta, alpha=scale)"
    cat = "        delta = torch.cat((w1_delta, w3_delta), dim=-1)\n"
    if expert.count(old) != 2 or expert.count(cat) != 1:
        raise RuntimeError(f"{target}: expert LoRA source drift")
    expert = expert.replace(old, "_AddIntoLoRADelta.apply(output, delta, scale)")
    expert = expert.replace(cat, cat + "        del w1_delta, w3_delta\n")
    source = source[:start] + HELPER + expert + source[end:]
    compile(source, str(target), "exec")
    target.write_text(source)
    print(MARKER + ": reuse expert delta buffers; preserve base output views")


if __name__ == "__main__":
    apply()
