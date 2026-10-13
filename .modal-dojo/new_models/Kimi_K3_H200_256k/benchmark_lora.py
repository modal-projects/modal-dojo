"""Test expert LoRA buffer reuse with the pinned TE kernels on one H200."""

import json
from pathlib import Path

import modal

app = modal.App("k3-lora-memory-proof")
root = Path(__file__).resolve().parents[3]
image = (
    modal.Image.from_registry("radixark/miles:dev-202609251434")
    .entrypoint([])
    .add_local_file(
        root / "tests/testdata/miles/k3_lora.py.delta_reuse.output",
        "/tmp/k3_lora_candidate.py",
    )
)


@app.function(image=image, gpu="H200", memory=65536, timeout=1200, serialized=True)
def benchmark():
    import gc
    import types

    import torch
    import transformer_engine.pytorch as te
    from torch.utils.checkpoint import checkpoint

    from miles_plugins.models.kimi_k3 import lora as original
    from miles_plugins.models.kimi_k3.ops import situ_and_mul

    candidate = types.ModuleType("k3_lora_candidate")
    exec(Path("/tmp/k3_lora_candidate.py").read_text(), candidate.__dict__)
    situ = torch.compile(situ_and_mul, fullgraph=True, dynamic=True)
    latent, intermediate, experts_count = 3584, 3072, 28

    class Linear(te.GroupedLinear):
        def forward(self, x, sizes):
            return super().forward(x, sizes), None

    def build(module, scale):
        experts = torch.nn.Module()
        experts.num_local_experts = experts_count
        for name, inp, out in (
            ("linear_fc1", latent, 2 * intermediate),
            ("linear_fc2", intermediate, latent),
        ):
            layer = Linear(
                experts_count,
                inp,
                out,
                bias=False,
                params_dtype=torch.bfloat16,
                init_method=lambda w: torch.nn.init.normal_(w, std=0.01),
            )
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
            setattr(experts, name, layer)
        moe = types.SimpleNamespace(
            experts=experts,
            config=types.SimpleNamespace(
                expert_tensor_parallel_size=1,
                moe_latent_size=latent,
                moe_ffn_hidden_size=intermediate,
            ),
        )
        module._apply_expert_lora(
            moe,
            types.SimpleNamespace(lora_rank=32),
            0,
            scale,
            0.0,
            include_fc2=True,
        )
        with torch.no_grad():
            for parameter in experts.lora_adapter.parameters():
                parameter.normal_(std=0.01)
        return experts

    def execute(model, x, grad, sizes, recompute, whole):
        model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_()

        def block(value):
            hidden, _ = model.linear_fc1(value, sizes)
            return model.linear_fc2(situ(hidden), sizes)[0]

        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        y = checkpoint(block, x, use_reentrant=True) if recompute else block(x)
        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated()
        y.backward(grad)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        indices = [0, len(x) // 2, len(x) - 1]
        values = {
            "output": (y if whole else y[indices]).detach().cpu(),
            "input_grad": (x.grad if whole else x.grad[indices]).detach().cpu(),
            **{
                name: p.grad.detach().cpu()
                for name, p in model.named_parameters()
                if p.requires_grad
            },
        }
        assert all(bool(torch.isfinite(v).all()) for v in values.values())
        assert any(
            bool(p.grad.abs().max() > 0) for p in model.lora_adapter.parameters()
        )
        stats = {
            "baseline_gib": baseline / 2**30,
            "forward_peak_extra_gib": (forward_peak - baseline) / 2**30,
            "forward_backward_peak_extra_gib": (peak - baseline) / 2**30,
        }
        del y, x
        model.zero_grad(set_to_none=True)
        return values, stats

    results = []
    for scale in (2.0, 0.37):
        torch.manual_seed(17)
        eager = build(original, scale)
        reuse = build(candidate, scale)
        reuse.load_state_dict(eager.state_dict())
        for large, recompute in ((False, False), (False, True), (True, False)):
            if large and scale != 2.0:
                continue
            sizes = (
                [14768] * experts_count
                if large
                else [16 * (i % 5) for i in range(experts_count)]
            )
            torch.manual_seed(23)
            x = torch.randn(sum(sizes), latent, device="cuda", dtype=torch.bfloat16)
            grad = torch.randn_like(x) * 0.01
            measurements = {}
            for name, model in (("eager", eager), ("reuse", reuse)):
                gc.collect()
                torch.cuda.empty_cache()
                values, stats = execute(model, x, grad, sizes, recompute, not large)
                measurements[name] = stats
                if name == "eager":
                    reference = values
                else:
                    for key, value in values.items():
                        torch.testing.assert_close(
                            value, reference[key], atol=0.02, rtol=0.02
                        )
                    measurements["max_output_abs"] = (
                        (values["output"] - reference["output"]).abs().max().item()
                    )
                    measurements["max_input_grad_abs"] = (
                        (values["input_grad"] - reference["input_grad"])
                        .abs()
                        .max()
                        .item()
                    )
                del values
            results.append(
                {
                    "scale": scale,
                    "rows": sum(sizes),
                    "recompute": recompute,
                    **measurements,
                }
            )
            print(json.dumps(results[-1]), flush=True)
            del x, grad, reference
        del eager, reuse
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "results": results,
    }


@app.local_entrypoint()
def main():
    result = benchmark.remote()
    Path(__file__).with_name("lora_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
