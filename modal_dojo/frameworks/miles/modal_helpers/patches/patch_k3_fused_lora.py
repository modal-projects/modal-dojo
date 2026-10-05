"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles_plugins/models/kimi_k3/lora.py
"""

from pathlib import Path

TARGET = Path("/root/miles/miles_plugins/models/kimi_k3/lora.py")
MARKER = "PATCHED_K3_FUSED_LORA"
HELPER_ANCHOR = "def _apply_expert_lora(\n"
HELPER = f"""# {MARKER}
@torch.compile(dynamic=True, fullgraph=True)
def _expert_lora_sum(output, w1_delta, w3_delta, scale):
    return torch.add(output, torch.cat((w1_delta, w3_delta), dim=-1), alpha=scale)


class _MergeExpertLoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, w1_delta, w3_delta, scale):
        ctx.width = w1_delta.shape[-1]
        ctx.scale = scale
        return _expert_lora_sum(output, w1_delta, w3_delta, scale)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output if ctx.needs_input_grad[0] else None,
            grad_output[..., :ctx.width] * ctx.scale if ctx.needs_input_grad[1] else None,
            grad_output[..., ctx.width:] * ctx.scale if ctx.needs_input_grad[2] else None,
            None,
        )


_merge_expert_lora = _MergeExpertLoRA.apply


class _MergeExpertLoRAFC2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, delta, scale):
        ctx.scale = scale
        ctx.mark_dirty(delta)
        return torch.add(output, delta, alpha=scale, out=delta)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output if ctx.needs_input_grad[0] else None,
            grad_output * ctx.scale if ctx.needs_input_grad[1] else None,
            None,
        )


_merge_expert_lora_fc2 = _MergeExpertLoRAFC2.apply


"""
ANCHOR = """        delta = torch.cat((w1_delta, w3_delta), dim=-1)
        return torch.add(output, delta, alpha=scale), bias
"""
REPLACEMENT = """        return _merge_expert_lora(output, w1_delta, w3_delta, scale), bias
"""
FC2_ANCHOR = """            delta = F.linear(inner, adapter.w2_lora_B)
            return torch.add(output, delta, alpha=scale), bias
"""
FC2_REPLACEMENT = """            delta = F.linear(inner, adapter.w2_lora_B)
            return _merge_expert_lora_fc2(output, delta, scale), bias
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if (
            source.count(HELPER) != 1
            or source.count(REPLACEMENT) != 1
            or source.count(FC2_REPLACEMENT) != 1
            or ANCHOR in source
            or FC2_ANCHOR in source
        ):
            raise RuntimeError(f"{target}: unexpected patched K3 LoRA source")
        return
    if (
        source.count(ANCHOR) != 1
        or source.count(HELPER_ANCHOR) != 1
        or source.count(FC2_ANCHOR) != 1
    ):
        raise RuntimeError(f"{target}: unexpected K3 LoRA source")
    source = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    source = source.replace(ANCHOR, REPLACEMENT)
    source = source.replace(FC2_ANCHOR, FC2_REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: fuse K3 expert LoRA concatenation and addition")


if __name__ == "__main__":
    apply()
