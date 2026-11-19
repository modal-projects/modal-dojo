"""Check the pinned Inkling backend before a full-model training proof.

CPU image/parser check:
  uv run modal run scripts/validate_inkling2_long_context.py
Single-B300 value/gradient check against the original attention kernel:
  uv run modal run scripts/validate_inkling2_long_context.py --gpu
Synthetic 256k attention/head memory stress, reserving 160 GB for base weights:
  uv run modal run scripts/validate_inkling2_long_context.py --stress

This is a component preflight, not evidence of full 256k training. A full-length
Dojo proof and smoke must also pass with real SWE rewards and dashboard timing.
"""

import json
from pathlib import Path

import modal

from modal_dojo import Inkling_2_Small_LoRA_256k_Recipe
from modal_dojo.frameworks.miles.launcher import _build_miles_base_image

app = modal.App("inkling-256k-image-preflight")
recipe = Inkling_2_Small_LoRA_256k_Recipe()
image = _build_miles_base_image(recipe).env(recipe.environment)


@app.function(image=image, serialized=True, timeout=600, cpu=4, memory=16384)
def validate():
    import subprocess
    import sys

    from modal_dojo.frameworks.miles import inkling2_long_context

    root = Path("/root/Megatron-LM/megatron/core")
    targets = [
        root / "transformer/transformer_block.py",
        Path("/root/miles/miles_plugins/models/inkling2/layers.py"),
        Path("/root/miles/miles/backends/training_utils/loss/hub/logit_processors.py"),
    ]
    for path in targets:
        source = path.read_text()
        assert "PATCHED_MODAL_DOJO_INKLING2_LONG_CONTEXT" in source, path
        compile(source, str(path), "exec")
    assert "output_processor" in (root / "models/gpt/gpt_model.py").read_text()
    assert (
        "PATCHED_MODAL_DOJO_GLM53_MOE_SEQ_CHUNK"
        in (root / "transformer/moe/moe_layer.py").read_text()
    )
    import torch

    x = torch.randn(17, 1, 8, requires_grad=True)
    weight = torch.randn(29, 8, requires_grad=True)
    tokens = torch.randint(0, 29, (1, 17))
    scores = inkling2_long_context.stream_token_scores(
        x,
        tokens,
        lambda h: h @ weight.T,
        lambda y: y,
        chunk_size=3,
    )
    scores.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()
    # Parser inspection does not instantiate the model or touch GPU memory.
    result = subprocess.run(
        [sys.executable, "/root/miles/train.py", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-6000:])
    for flag in (
        "--max-seq-len",
        "--rollout-max-prompt-len",
        "--custom-model-provider-path",
    ):
        assert flag in result.stdout, flag
    return {
        "torch": str(torch.__version__),
        "patches": "applied",
        "streamed_cpu_backward": "passed",
        "parser": "help passed",
    }


@app.local_entrypoint()
def main(gpu: bool = False, stress: bool = False):
    if gpu and stress:
        raise ValueError("Choose either --gpu or --stress")
    result = (
        stress_gpu.remote()
        if stress
        else validate_gpu.remote()
        if gpu
        else validate.remote()
    )
    print(json.dumps(result, indent=2))
    name = (
        "component_stress" if stress else "gpu_preflight" if gpu else "image_preflight"
    )
    directory = Path(".modal-dojo/new_models/Inkling_2_Small")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(result, indent=2))


@app.function(
    image=image, serialized=True, gpu="B300", timeout=1800, cpu=8, memory=65536
)
def validate_gpu():
    import torch
    from modal_dojo.frameworks.miles import inkling2_long_context as lc
    from miles_plugins.models.inkling2.ops import relative_attention as original

    results = {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name()}
    for window in (None, 31):
        torch.manual_seed(19)
        q = torch.randn(
            384, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            384, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn_like(k, requires_grad=True)
        r = torch.randn(
            384, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        projection = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            * 0.1
        )
        inputs = (q, k, v, r, projection)
        expected, expected_lse = original(
            q,
            k,
            v,
            torch.einsum("thd,de->the", r, projection),
            [157, 227],
            window=window,
            block_mask_cache={},
        )
        expected_grads = torch.autograd.grad(
            expected.float().square().mean() + expected_lse.mean(), inputs
        )
        actual, actual_lse = lc.relative_attention(
            *inputs, [157, 227], window=window, block_mask_cache={}, chunk_size=128
        )
        grads = torch.autograd.grad(
            actual.float().square().mean() + actual_lse.mean(), inputs
        )
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.005, rtol=0.005)
        for g, e in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(g, e, atol=1e-4, rtol=0.03)
        results[f"attention_{window}"] = {
            "max_output_diff": (actual - expected).abs().max().item(),
            "max_grad_diff": max(
                (g - e).abs().max().item() for g, e in zip(grads, expected_grads)
            ),
        }
    _check_native_checkpoint()
    results["native_checkpoint_backward"] = "passed"
    return results


def _check_native_checkpoint():
    import torch
    from modal_dojo.frameworks.miles import inkling2_long_context as lc

    # Exercise the actual Megatron checkpoint, including its reentrant backward,
    # rather than only the torch stand-in used by local CPU tests.
    hidden = torch.randn(17, 1, 8, device="cuda", requires_grad=True)
    context = torch.randn(17, 1, 2, 8, device="cuda", requires_grad=True)

    def layer(h, mask, c):
        return (h + c.sum(-2)).square()

    expected = torch.autograd.grad(
        layer(hidden, None, context).sum(), (hidden, context)
    )
    token = lc._checkpoint_store.set(lc.CheckpointStore())
    try:
        output = lc.activation_checkpoint(layer, False, hidden, None, context)
    finally:
        lc._checkpoint_store.reset(token)
    output.sum().backward()
    for actual, reference in zip((hidden.grad, context.grad), expected, strict=True):
        torch.testing.assert_close(actual, reference)


@app.function(
    image=image, serialized=True, gpu="B300", timeout=1800, cpu=8, memory=65536
)
def stress_gpu():
    import gc
    import time

    import torch
    import torch.nn.functional as F

    from modal_dojo.frameworks.miles import inkling2_long_context as lc

    _check_native_checkpoint()
    torch.manual_seed(31)
    # Component test only. The reserve is not a loaded model and does not
    # reproduce its other live activations, optimizer, routing or collectives.
    reserve = torch.empty(160_000_000_000, dtype=torch.uint8, device="cuda")
    results = {
        "tokens": 262144,
        "reserved_bytes": reserve.numel(),
        "native_checkpoint_backward": "passed",
        "gpu": torch.cuda.get_device_name(),
    }

    def attention():
        def tensor(*shape):
            return torch.randn(
                *shape, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )

        q, k, v = (
            tensor(262144, 48, 128),
            tensor(262144, 8, 128),
            tensor(262144, 8, 128),
        )
        r, projection = tensor(262144, 48, 16), tensor(16, 1024)
        out, lse = lc.relative_attention(
            q, k, v, r, projection, [262144], window=None, block_mask_cache={}
        )
        (out.float().square().mean() + lse.mean()).backward()
        assert out.shape == q.shape and torch.isfinite(out).all()
        assert all(torch.isfinite(x.grad).all() for x in (q, k, v, r, projection))

    def head():
        hidden = torch.randn(
            262144, 1, 4608, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(201024, 4608, device="cuda", dtype=torch.bfloat16) * 0.01
        a = torch.randn(
            32, 4608, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        b = torch.zeros(
            201024, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        tokens = torch.randint(0, 200058, (1, 262144), device="cuda")

        def project(h):
            return (
                F.linear(h.float(), weight.float())
                + F.linear(F.linear(h, a), b).float()
            )

        scores = lc.stream_token_scores(
            hidden,
            tokens,
            project,
            lambda x: 30 * torch.tanh(x / 30),
            vocab_size=200058,
        )
        (scores[..., 0].mean() + 0.01 * scores[..., 1].mean()).backward()
        assert scores.shape == (1, 262144, 2) and torch.isfinite(scores).all()
        assert all(torch.isfinite(x.grad).all() for x in (hidden, a, b))

    for name, check in (("full_attention", attention), ("streamed_head", head)):
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        check()
        torch.cuda.synchronize()
        results[name] = {
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        print(name, results[name], flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return results
