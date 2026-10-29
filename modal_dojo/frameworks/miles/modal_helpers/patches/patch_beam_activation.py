"""
image: radixark/miles:v0.1.2
file: megatron/core/transformer/moe/experts.py

Chunk the unfused BF16 SwiGLU forward and backward temporaries.
The input, output, and gradients still span the full local token batch.

TODO: Remove once the pinned Megatron bounds expert activation temporaries
with equivalent BF16 rounding and supports Bridge LoRA, CP2, and full recompute.
The open chunked-EP implementation is related, not a drop-in replacement:
it requires MXFP8/CP1 and rejects full recompute.
https://github.com/NVIDIA/Megatron-LM/pull/7331
"""

from pathlib import Path

HELPER = '''class _BeamChunkedSwiGLU(torch.autograd.Function):
    """Bound temporaries while preserving Beam's eager BF16 rounding points."""

    @staticmethod
    def _activation(x, probs):
        gate, value = x.chunk(2, dim=-1)
        activated = F.silu(gate) * (value + 0.0)
        return (activated * probs).to(x.dtype)

    @staticmethod
    def forward(ctx, x, probs):
        ctx.save_for_backward(x, probs)
        output = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
        for start in range(0, x.shape[0], 8192):
            end = start + 8192
            output[start:end].copy_(_BeamChunkedSwiGLU._activation(x[start:end], probs[start:end]))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, probs = ctx.saved_tensors
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dp = torch.empty_like(probs) if ctx.needs_input_grad[1] else None
        for start in range(0, x.shape[0], 8192):
            end = start + 8192
            with torch.enable_grad():
                xc = x[start:end].detach().requires_grad_(True)
                pc = probs[start:end].detach().requires_grad_(True)
                y = _BeamChunkedSwiGLU._activation(xc, pc)
                gx, gp = torch.autograd.grad(y, (xc, pc), grad_output[start:end])
            if dx is not None:
                dx[start:end].copy_(gx)
            if dp is not None:
                dp[start:end].copy_(gp)
        return dx, dp


'''
ANCHOR = """        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
"""
REPLACEMENT = """        def bias_act_func(intermediate_parallel, bias_parallel, permuted_probs):
            # MODAL_DOJO_BEAM_CHUNKED_ACTIVATION: preserve unfused model numerics.
            if (
                intermediate_parallel.dtype == torch.bfloat16
                and intermediate_parallel.ndim == 2
                and bias_parallel is None
                and permuted_probs is not None
                and self.activation_func == F.silu
                and self.config.gated_linear_unit
                and self.config.gated_activation_func is None
                and self.config.glu_linear_offset == 0.0
                and self.config.activation_func_clamp_value is None
                and self.config.moe_mlp_glu_interleave_size is None
                and not self.config.moe_activation_in_fp32
                and not self.config.use_te_activation_func
                and not self.config.bias_activation_fusion
            ):
                return _BeamChunkedSwiGLU.apply(intermediate_parallel, permuted_probs)
"""


def patch_source(source: str) -> str:
    if HELPER in source and REPLACEMENT in source:
        return source
    marker = "class TEGroupedMLP(MegatronModule):"
    if source.count(ANCHOR) != 1 or source.count(marker) != 1:
        raise RuntimeError("Pinned expert activation implementation changed")
    return source.replace(marker, HELPER + marker).replace(ANCHOR, REPLACEMENT)


def main() -> None:
    path = Path("/root/Megatron-LM/megatron/core/transformer/moe/experts.py")
    path.write_text(patch_source(path.read_text()))
    print(f"Applied Beam activation patch to {path}")


if __name__ == "__main__":
    main()
