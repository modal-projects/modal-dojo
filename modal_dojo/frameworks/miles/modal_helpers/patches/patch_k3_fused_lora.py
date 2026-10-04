"""Fuse routed-expert LoRA concatenation and addition on the pinned K3 image.

The eager path materializes a full-width concatenated delta before allocating
the addition result. Inductor writes the sum directly, preserving autograd and
leaving Transformer Engine's view output unmodified.
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
        # No activation values are needed for this linear operation. Avoid
        # an extra full-width contiguous gradient in compiled cat backward.
        return (
            grad_output if ctx.needs_input_grad[0] else None,
            grad_output[..., :ctx.width] * ctx.scale if ctx.needs_input_grad[1] else None,
            grad_output[..., ctx.width:] * ctx.scale if ctx.needs_input_grad[2] else None,
            None,
        )


_merge_expert_lora = _MergeExpertLoRA.apply


"""
ANCHOR = """        delta = torch.cat((w1_delta, w3_delta), dim=-1)
        return torch.add(output, delta, alpha=scale), bias
"""
REPLACEMENT = """        return _merge_expert_lora(output, w1_delta, w3_delta, scale), bias
"""


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        if (
            source.count(HELPER) != 1
            or source.count(REPLACEMENT) != 1
            or ANCHOR in source
        ):
            raise RuntimeError(f"{target}: unexpected patched K3 LoRA source")
        return
    if source.count(ANCHOR) != 1 or source.count(HELPER_ANCHOR) != 1:
        raise RuntimeError(f"{target}: unexpected K3 LoRA source")
    source = source.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"Patched {target}: fuse K3 expert LoRA concatenation and addition")


if __name__ == "__main__":
    apply()
