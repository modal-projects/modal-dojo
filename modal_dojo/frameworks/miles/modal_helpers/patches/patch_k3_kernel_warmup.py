"""
image: radixark/miles:dev-202609251434
commit: https://github.com/radixark/miles/commit/41c5e38b94ea23677de93b01a4a77d55677a8f09
file: miles/backends/megatron_utils/actor.py::MegatronTrainRayActor.init
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/backends/megatron_utils/actor.py")
MARKER = "PATCHED_K3_KERNEL_WARMUP"
ANCHOR = "        verify_megatron_parallel_state(self.model)\n"
REPLACEMENT = f"""        verify_megatron_parallel_state(self.model)
        # {MARKER}
        if role == "actor":
            from .k3_kernel_warmup import warm_k3_kernels

            warm_k3_kernels(args, self.model)
"""
MODULE = '''"""Warm KDA kernels on every pipeline stage before the first microbatch."""
import os
import time

import torch


def warm_k3_kernels(args, models):
    if os.environ.get("DOJO_K3_KERNEL_WARMUP") != "1":
        return
    if getattr(args, "model_name", None) != "kimi_k3":
        return
    from miles_plugins.models.kimi_k3.ops import kda
    from fla.ops.cp import build_cp_context

    seen = set()
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank()
    # No model forward, optimizer step or model gradients: use disposable inputs
    # and restore RNG state so enabling warmup does not alter sampled training.
    with torch.random.fork_rng(devices=[device]), torch.enable_grad():
        for model in models:
            for layer in model.modules():
                if not getattr(layer, "is_kda", False):
                    continue
                cp_size = layer.cp_size
                tokens = (args.seq_length + cp_size - 1) // cp_size
                tokens = min(tokens, args.max_tokens_per_gpu)
                key = (tokens, layer.local_num_heads, layer.head_dim, cp_size,
                       layer.gate_lower_bound, layer.config.params_dtype)
                if key in seen:
                    continue
                seen.add(key)
                started = time.monotonic()
                print(f"DOJO_K3_WARMUP rank={rank} phase=start shape={key}", flush=True)
                shape = (1, tokens, layer.local_num_heads, layer.head_dim)
                q, k, v, g = [torch.randn(shape, device=device,
                    dtype=layer.config.params_dtype, requires_grad=True) for _ in range(4)]
                beta = torch.rand(shape[:-1], device=device, dtype=torch.float32, requires_grad=True)
                a = torch.zeros(layer.local_num_heads, device=device, dtype=torch.float32, requires_grad=True)
                bias = torch.zeros(layer.local_num_heads * layer.head_dim,
                                   device=device, dtype=torch.float32, requires_grad=True)
                cu = torch.tensor([0, tokens * cp_size], device=device, dtype=torch.int32)
                cp = build_cp_context(cu, layer.cp_group) if cp_size > 1 else None
                output = kda(q, k, v, g, beta, a, bias, layer.gate_lower_bound,
                             cu_seqlens=None if cp is not None else cu, cp_context=cp)
                torch.cuda.synchronize()
                forward_s = time.monotonic() - started
                tick = time.monotonic()
                torch.autograd.grad(output.float().square().mean(), (q, k, v, g, beta, a, bias))
                torch.cuda.synchronize()
                print(f"DOJO_K3_WARMUP rank={rank} phase=end forward_s={forward_s:.3f} "
                      f"backward_s={time.monotonic() - tick:.3f}", flush=True)
                del output, q, k, v, g, beta, a, bias, cu, cp
'''


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    helper = target.with_name("k3_kernel_warmup.py")
    if MARKER in source:
        if source.count(REPLACEMENT) != 1 or helper.read_text() != MODULE:
            raise RuntimeError(f"{target}: unexpected patched K3 warmup source")
        return
    if source.count(ANCHOR) != 1:
        raise RuntimeError(f"{target}: expected one parallel-state verification")
    source = source.replace(ANCHOR, REPLACEMENT)
    compile(source, str(target), "exec")
    compile(MODULE, str(helper), "exec")
    helper.write_text(MODULE)
    target.write_text(source)


if __name__ == "__main__":
    apply()
