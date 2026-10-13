"""Check offloaded outer checkpoints against the pinned Megatron CUDA runtime."""

import modal

from modal_dojo import GLM_5_3_Flash_LoRA_256k_Recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_checkpoint_cpu as patch,
)

app = modal.App("glm256k-checkpoint-cpu-check")
image = modal.Image.from_registry(GLM_5_3_Flash_LoRA_256k_Recipe().docker_image)


@app.function(image=image, gpu="H200", timeout=300, serialized=True)
def check(helper_source: str):
    import gc

    import torch
    from megatron.core import tensor_parallel

    namespace = {"torch": torch, "tensor_parallel": tensor_parallel}
    exec(helper_source, namespace)
    offloaded = namespace["_modal_dojo_cpu_checkpoint"]

    def run(use_cpu):
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(23)
        x = torch.randn(4096, 1024, device="cuda", requires_grad=True)
        weights = [
            torch.randn(1024, device="cuda", requires_grad=True) for _ in range(6)
        ]
        start = torch.cuda.memory_allocated()
        h = x
        for weight in weights:

            def layer(value, w=weight):
                # Also exercise the nested checkpoint used by chunked MoE.
                return tensor_parallel.checkpoint(
                    lambda t: torch.nn.functional.dropout(t.sin() * w, p=0.1),
                    False,
                    value,
                )

            checkpoint = offloaded if use_cpu else tensor_parallel.checkpoint
            h = checkpoint(layer, False, h)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - start
        value = h.detach().cpu()
        h.square().mean().backward()
        grads = [x.grad.cpu(), *(w.grad.cpu() for w in weights)]
        return value, grads, held

    expected, expected_grads, baseline = run(False)
    actual, actual_grads, offload = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert offload < baseline / 2, (baseline, offload)
    return {
        "forward_and_gradients": "exact",
        "baseline_held_bytes": baseline,
        "offloaded_held_bytes": offload,
    }


@app.local_entrypoint()
def main():
    print(check.remote(patch.APPENDIX))
