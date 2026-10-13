"""Exercise the pinned mHC MLP path and its manager lifetime on one GPU."""

from pathlib import Path

import modal

from modal_dojo import GLM_5_3_Flash_LoRA_256k_Recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_checkpoint_cpu as cpu,
    patch_glm_5_3_flash_mhc_release as release,
)

app = modal.App("glm256k-mhc-release-check")
image = modal.Image.from_registry(GLM_5_3_Flash_LoRA_256k_Recipe().docker_image)


@app.function(image=image, gpu="H100", timeout=600, serialized=True)
def check(cpu_source: str, patch_source: str):
    import gc
    import inspect
    import textwrap

    import torch
    from megatron.core import tensor_parallel
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )
    from megatron.core.transformer import transformer_layer as module
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    namespace = {"torch": torch, "tensor_parallel": tensor_parallel}
    exec(cpu_source, namespace)
    checkpoint = namespace["_modal_dojo_cpu_checkpoint"]
    patch_ns = {"__name__": "patch"}
    exec(patch_source, patch_ns)
    original = module.HyperConnectionTransformerLayer._forward_post_mlp_with_fused_hyper_connection
    patched_source = patch_ns["patch_source"](inspect.getsource(original))
    # Dedent the method separately from the file marker.
    patched_source = textwrap.dedent(patched_source.split("\n", 1)[1])
    method_ns = dict(vars(module))
    exec(patched_source, method_ns)
    patched = method_ns[original.__name__]

    config = TransformerConfig(
        num_layers=12,
        hidden_size=128,
        num_attention_heads=1,
        enable_hyper_connections=True,
        num_residual_streams=4,
        use_fused_mhc=False,
        params_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(128, device="cuda") * 0.05)

        def forward(self, h, **kwargs):
            return tensor_parallel.checkpoint(
                lambda x: x.tanh() * self.weight, False, h
            ), None

    class Layer(module.HyperConnectionTransformerLayer):
        def __init__(self, index, clear):
            torch.nn.Module.__init__(self)
            self.config = config
            self.layer_number = index + 1
            self.mlp_hyper_connection = HyperConnectionModule(
                config, self.layer_number
            ).cuda()
            self.mlp_hyper_connection.requires_grad_(False)
            self.is_moe_layer = False
            self.recompute_pre_mlp_layernorm = False
            self.recompute_mlp = False
            self.offload_mlp_norm = False
            self.off_interface = FineGrainedActivationOffloadingInterface
            self.pre_mlp_layernorm = torch.nn.Identity()
            self.mlp = MLP()
            self.hidden_dropout = 0.0
            self.bias_dropout_add_exec_handler = torch.enable_grad
            self._forward_post_mlp_with_fused_hyper_connection = (
                patched if clear else original
            ).__get__(self, Layer)

        def forward(self, h):
            return self._forward_mlp(h)

    def run(clear):
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(43)
        layers = [Layer(i, clear) for i in range(12)]
        x = torch.randn(4096, 1, 512, device="cuda", requires_grad=True)
        h = x
        memory = []
        for layer in layers:

            def forward(h, layer=layer):
                memory.append(
                    torch.cuda.memory_allocated()
                ) if torch.is_grad_enabled() else None
                return layer(h)

            h = checkpoint(forward, False, h)
        output = h.detach().cpu()
        h.square().mean().backward()
        torch.cuda.synchronize()
        grads = [x.grad.cpu(), *(layer.mlp.weight.grad.cpu() for layer in layers)]
        held = torch.cuda.memory_allocated()
        managers = sum(layer.mlp_norm_manager is not None for layer in layers)
        for layer in layers:
            layer.mlp_norm_manager = None
        del h, x, layers, layer, forward
        gc.collect()
        return output, grads, held, managers, memory

    a, ag, am, ac, atrend = run(False)
    b, bg, bm, bc, btrend = run(True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for x, y in zip(ag, bg, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert ac == 12 and bc == 0, (ac, bc)
    assert bm < am - 150 * 1024**2, (am, bm)
    return {
        "forward_and_gradients": "exact",
        "retained_bytes_before": am,
        "retained_bytes_after": bm,
        "managers_before": ac,
        "managers_after": bc,
        "backward_entry_bytes_before": atrend,
        "backward_entry_bytes_after": btrend,
    }


@app.local_entrypoint()
def main():
    print(check.remote(cpu.APPENDIX, Path(release.__file__).read_text()))
