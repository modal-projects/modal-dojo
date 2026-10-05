"""Single-B300 compact MXFP4 shard and LoRA numerical proof.

Compares the same weights/activations with 256-wide padded and 192-wide compact
storage. This is a kernel proof, not full-engine memory or throughput validation.
"""

import json
from pathlib import Path
import modal

ROOT = Path(".modal-dojo/new_models/Kimi_K3_B300")
app = modal.App("k3-b300-compact-marlin-proof")
image = modal.Image.from_registry("radixark/miles:dev-202609251434").entrypoint([])


@app.function(
    image=image,
    gpu="B300",
    cpu=8,
    memory=32768,
    timeout=1200,
    retries=0,
    serialized=True,
)
def probe():
    import time
    import traceback
    from types import SimpleNamespace as NS
    import torch
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
        prepare_moe_nvfp4_layer_for_marlin,
    )
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.lora.lora_moe_runner_marlin import MarlinLoraRunnerCore

    torch.manual_seed(42)
    E, H, intermediate, R = 4, 3584, 192, 32
    dev = "cuda"
    dtype = torch.bfloat16
    config = NS(
        is_gated=True,
        activation="situ",
        routed_scaling_factor=1.0,
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
    )
    raw13 = torch.randint(
        0, 256, (E, 2 * intermediate, H // 2), device=dev, dtype=torch.uint8
    )
    raw2 = torch.randint(
        0, 256, (E, H, intermediate // 2), device=dev, dtype=torch.uint8
    )
    # Identical nonzero low-rank updates on padded and unpadded layouts.
    a13 = torch.randn(E, H, R, device=dev, dtype=dtype) * 0.02
    b13 = torch.randn(E, R, 2 * intermediate, device=dev, dtype=dtype) * 0.02
    a2 = torch.randn(E, intermediate, R, device=dev, dtype=dtype) * 0.02
    b2 = torch.randn(E, R, H, device=dev, dtype=dtype) * 0.02
    outputs = {}
    report = {
        "gpu": torch.cuda.get_device_name(),
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "tests": [],
        "passed": False,
    }
    started = time.monotonic()
    for fmt, width in [("mxfp4", 256), ("mxfp4", 192)]:
        case = {"format": fmt, "width": width}
        try:
            layer = torch.nn.Module()
            layer.orig_dtype = layer.params_dtype = dtype
            layer.moe_runner_config = config
            layer.quant_config = NS(group_size=16)
            layer.intermediate_size_per_partition = width
            w13 = torch.zeros(E, 2, width, H // 2, device=dev, dtype=torch.uint8)
            w13[:, :, :intermediate] = raw13.reshape(E, 2, intermediate, H // 2)
            w2 = torch.zeros(E, H, width // 2, device=dev, dtype=torch.uint8)
            w2[:, :, : intermediate // 2] = raw2

            def param(name, value):
                layer.register_parameter(
                    name, torch.nn.Parameter(value, requires_grad=False)
                )

            param("w13_weight", w13.reshape(E, 2 * width, H // 2))
            param("w2_weight", w2)
            group = 32 if fmt == "mxfp4" else 16

            def scales(shape):
                if fmt == "mxfp4":
                    return torch.full(shape, 122, device=dev, dtype=torch.uint8)
                return torch.full(shape, 0.03125, device=dev, dtype=dtype)

            param("w13_weight_scale", scales((E, 2 * width, H // group)))
            param("w2_weight_scale", scales((E, H, width // group)))
            if fmt == "nvfp4":
                param("w13_weight_scale_2", torch.ones(E, device=dev, dtype=dtype))
                param("w2_weight_scale_2", torch.ones(E, device=dev, dtype=dtype))
                prepare_moe_nvfp4_layer_for_marlin(layer)
            else:
                prepare_moe_mxfp4_layer_for_marlin(layer)
            quant = MarlinMoeQuantInfo(
                w13_qweight=layer.w13_weight,
                w2_qweight=layer.w2_weight,
                w13_scales=layer.w13_weight_scale,
                w2_scales=layer.w2_weight_scale,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=4,
                w13_global_scale=getattr(layer, "w13_weight_scale_2", None),
                w2_global_scale=getattr(layer, "w2_weight_scale_2", None),
            )
            case["parameter_bytes"] = sum(
                p.numel() * p.element_size() for p in layer.parameters()
            )
            case["checks"] = []
            for M in (1, 16, 128):
                torch.manual_seed(100 + M)
                x = torch.randn(M, H, device=dev, dtype=dtype) * 0.02
                ids = torch.stack(
                    (
                        torch.arange(M, device=dev) % E,
                        (torch.arange(M, device=dev) + 1) % E,
                    ),
                    dim=1,
                ).int()
                weights = torch.full((M, 2), 0.5, device=dev, dtype=torch.float32)
                dispatch = NS(
                    hidden_states=x,
                    topk_output=NS(
                        topk_ids=ids,
                        topk_weights=weights,
                        router_logits=torch.zeros(M, E, device=dev),
                    ),
                )

                def after_gate(x, y, topk_weights, topk_ids):
                    for t in range(2):
                        for e in range(E):
                            mask = topk_ids[:, t] == e
                            delta = (x[mask] @ a13[e]) @ b13[e]
                            y[mask, t, :intermediate] += delta[:, :intermediate]
                            y[mask, t, width : width + intermediate] += delta[
                                :, intermediate:
                            ]

                def after_down(x, y, topk_weights, topk_ids):
                    x = x.reshape(M, 2, width)
                    for t in range(2):
                        for e in range(E):
                            mask = topk_ids[:, t] == e
                            delta = (x[mask, t, :intermediate] @ a2[e]) @ b2[e]
                            y[mask, t] += delta * topk_weights[mask, t, None].to(dtype)

                runner = MarlinLoraRunnerCore(config)
                base = runner.run_from_dispatch(
                    dispatch, quant, config, NS(after_gate_up=None, after_down=None)
                ).hidden_states
                output = runner.run_from_dispatch(
                    dispatch,
                    quant,
                    config,
                    NS(after_gate_up=after_gate, after_down=after_down),
                ).hidden_states
                torch.cuda.synchronize()
                assert torch.isfinite(output).all()
                assert not torch.equal(base, output), "LoRA hooks had no effect"
                key = ("mxfp4", 256, M)
                if key in outputs:
                    reference = outputs[key]
                    relative_l2 = (
                        (output.float() - reference.float()).norm()
                        / reference.float().norm()
                    ).item()
                    assert relative_l2 < 0.025, relative_l2
                else:
                    relative_l2 = 0.0
                outputs[(fmt, width, M)] = output.detach().clone()
                case["checks"].append(
                    {
                        "tokens": M,
                        "relative_l2_vs_padded": relative_l2,
                        "lora_delta_l2": (output.float() - base.float()).norm().item(),
                    }
                )
            case["passed"] = True
        except Exception:
            case.update(passed=False, error=traceback.format_exc())
        report["tests"].append(case)
        print(json.dumps(case), flush=True)
    report["passed"] = all(t["passed"] for t in report["tests"])
    report["seconds"] = time.monotonic() - started
    return report


@app.local_entrypoint()
def main():
    result = probe.remote()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "compact-marlin-proof.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("Compact Marlin proof failed; do not launch full validation")
