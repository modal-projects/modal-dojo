"""Compare exact SiTU math with torch.compile on one H200, without model weights."""

import json
from pathlib import Path

import modal

app = modal.App("k3-situ-memory-proof")
image = modal.Image.from_registry("radixark/miles:dev-202609251434").entrypoint([])


@app.function(image=image, gpu="H200", memory=32768, timeout=900, serialized=True)
def benchmark():
    import gc
    import time

    import torch

    from miles_plugins.models.kimi_k3.ops import situ_and_mul

    compiled = torch.compile(situ_and_mul, fullgraph=True, dynamic=True)
    correctness = []
    for dtype in (torch.float32, torch.bfloat16):
        for scale in (0.01, 1.0, 10.0, 100.0):
            torch.manual_seed(11)
            x = (
                torch.randn(513, 7168, device="cuda", dtype=dtype) * scale
            ).requires_grad_()
            grad = torch.randn(513, 3584, device="cuda", dtype=dtype)
            y = situ_and_mul(x)
            (expected_grad,) = torch.autograd.grad(y, x, grad)
            actual = compiled(x)
            (actual_grad,) = torch.autograd.grad(actual, x, grad)
            torch.testing.assert_close(
                actual,
                y,
                atol=1e-5 if dtype == torch.float32 else 0.008,
                rtol=1e-5 if dtype == torch.float32 else 0.008,
            )
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                atol=1e-4 if dtype == torch.float32 else 0.02,
                rtol=1e-4 if dtype == torch.float32 else 0.02,
            )
            correctness.append(
                {
                    "dtype": str(dtype),
                    "scale": scale,
                    "output_max_abs": (actual - y).abs().max().item(),
                    "grad_max_abs": (actual_grad - expected_grad).abs().max().item(),
                }
            )
            del x, grad, y, expected_grad, actual, actual_grad
    results = []
    # Synthetic width: large FP32 halves are 4.92 GiB, close to the failed
    # 4.93 GiB allocation. K3's actual expert intermediate width is 3072;
    # this test matches allocation size, not the exact matrix aspect ratio.
    for rows in (131072, 368640):
        for name, fn in (("eager", situ_and_mul), ("compiled", compiled)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(17)
            x = torch.randn(
                rows, 7168, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            grad = torch.ones(rows, 3584, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            y = fn(x)
            torch.cuda.synchronize()
            forward_seconds = time.monotonic() - started
            forward_peak = torch.cuda.max_memory_allocated()
            forward_live = torch.cuda.memory_allocated()
            started = time.monotonic()
            (dx,) = torch.autograd.grad(y, x, grad)
            torch.cuda.synchronize()
            results.append(
                {
                    "mode": name,
                    "rows": rows,
                    "baseline_gib": baseline / 2**30,
                    "forward_peak_extra_gib": (forward_peak - baseline) / 2**30,
                    "forward_live_extra_gib": (forward_live - baseline) / 2**30,
                    "forward_backward_peak_extra_gib": (
                        torch.cuda.max_memory_allocated() - baseline
                    )
                    / 2**30,
                    "forward_seconds": forward_seconds,
                    "backward_seconds": time.monotonic() - started,
                    "finite": bool(torch.isfinite(dx).all()),
                }
            )
            print(json.dumps(results[-1]), flush=True)
            indices = [0, rows // 2, rows - 1]
            if name == "eager":
                reference_y = y[indices].detach().cpu()
                reference_dx = dx[indices].detach().cpu()
            else:
                torch.testing.assert_close(
                    y[indices].detach().cpu(), reference_y, atol=0.008, rtol=0.008
                )
                torch.testing.assert_close(
                    dx[indices].detach().cpu(), reference_dx, atol=0.02, rtol=0.02
                )
            del x, grad, y, dx
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "correctness": correctness,
        "memory": results,
    }


@app.local_entrypoint()
def main():
    result = benchmark.remote()
    Path(__file__).with_name("situ_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
