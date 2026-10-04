"""Reproduce FP8 checkpoint OOM and prove the H200 patch on one GPU.

Run: uv run modal run --env <env> --detach scripts/validate_k3_fp8_checkpoint.py

Uses synthetic weights and a local checkpoint; no model download or shared
volume is needed. This proves loader allocation behavior, exact BF16 loading,
FP8 copy-back, and frozen initialization-copy release. It does not validate
multi-rank resharding or full-model training memory.
"""

import json
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("k3-h200-fp8-checkpoint-regression")
image = (
    modal.Image.from_registry("radixark/miles:dev-202609251434")
    .entrypoint([])
    .env({"PYTHONPATH": "/root/Megatron-LM:/root/miles"})
    .add_local_file(
        ROOT
        / "modal_dojo/frameworks/miles/modal_helpers/patches/patch_k3_fp8_checkpoint.py",
        "/root/fp8_checkpoint_patch.py",
    )
)


@app.function(
    image=image, gpu="H200", memory=65536, cpu=8, serialized=True, timeout=900
)
def check():
    import gc
    import importlib
    import importlib.util
    import torch
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    from torch import nn
    from megatron.core.dist_checkpointing import serialization
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    dist.init_process_group(
        "gloo", init_method="file:///tmp/checkpoint-test-store", rank=0, world_size=1
    )
    torch.manual_seed(31)
    with te.fp8_model_init(enabled=True, preserve_high_precision_init_val=True):
        model = nn.ModuleDict(
            {
                "fc1": te.GroupedLinear(
                    4,
                    3584,
                    6144,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    device="cuda",
                ),
                "fc2": te.GroupedLinear(
                    4,
                    3072,
                    3584,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    device="cuda",
                ),
            }
        )
    for p in model.parameters():
        p.requires_grad_(False)
    del p
    weights = dict(model.named_parameters())
    expected = {
        name: torch.full(tuple(p.shape), 0.01 + i * 0.003, dtype=torch.bfloat16)
        for i, (name, p) in enumerate(weights.items())
    }
    checkpoint = Path("/tmp/synthetic-bf16-checkpoint")
    checkpoint.mkdir()
    serialization.save(
        {n: ShardedTensor.from_rank_offsets(n, t) for n, t in expected.items()},
        checkpoint,
    )

    def template():
        return {n: ShardedTensor.from_rank_offsets(n, p) for n, p in weights.items()}

    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    budget = 96 * 2**20
    torch.cuda.set_per_process_memory_fraction((base + budget) / total)
    result = {"allocation_budget_mib": 96, "parameter_mib": base / 2**20}
    try:
        old = serialization.load(template(), checkpoint)
        result["original"] = "unexpected_pass"
        del old
    except torch.OutOfMemoryError as exc:
        result["original"] = "oom"
        result["original_error"] = str(exc)
    # Leave the except block to release traceback references to old templates.
    gc.collect()
    torch.cuda.empty_cache()
    spec = importlib.util.spec_from_file_location(
        "fp8_checkpoint_patch", "/root/fp8_checkpoint_patch.py"
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.apply()
    serialization = importlib.reload(serialization)
    torch.cuda.reset_peak_memory_stats()
    loaded = serialization.load(template(), checkpoint)
    torch.cuda.synchronize()
    result["patched_peak_extra_mib"] = (
        torch.cuda.max_memory_allocated() - base
    ) / 2**20
    result["loaded_on_cpu"] = all(t.device.type == "cpu" for t in loaded.values())
    for n, t in expected.items():
        torch.testing.assert_close(loaded[n], t, rtol=0, atol=0)
    result["loaded_exact_bf16"] = True
    torch.cuda.set_per_process_memory_fraction(1.0)
    with torch.no_grad():
        for name, p in weights.items():
            p.copy_(loaded[name])
    result["max_abs_quantization_error"] = max(
        (p.dequantize().cpu().float() - expected[n].float()).abs().max().item()
        for n, p in weights.items()
    )
    before = {n: p.dequantize().cpu() for n, p in weights.items()}
    result["cpu_init_before_mib"] = (
        sum(
            p._high_precision_init_val.numel()
            * p._high_precision_init_val.element_size()
            for p in weights.values()
            if isinstance(getattr(p, "_high_precision_init_val", None), torch.Tensor)
        )
        / 2**20
    )
    # The exact loop installed in native K3 LoRA runs before adapter injection.
    code = patch.LORA_REPLACEMENT.split(
        "    _enable_full_recompute_input_grads(model)"
    )[0]
    import textwrap

    exec(textwrap.dedent(code), {"model": model})
    result["cpu_init_after_mib"] = (
        sum(
            p._high_precision_init_val.numel()
            * p._high_precision_init_val.element_size()
            for p in weights.values()
            if isinstance(getattr(p, "_high_precision_init_val", None), torch.Tensor)
        )
        / 2**20
    )
    for n, p in weights.items():
        torch.testing.assert_close(p.dequantize().cpu(), before[n], rtol=0, atol=0)
    result["init_release_preserves_weights"] = True
    assert (
        result["original"] == "oom"
        and result["loaded_on_cpu"]
        and result["loaded_exact_bf16"]
    )
    assert result["patched_peak_extra_mib"] < 96
    assert result["max_abs_quantization_error"] < 0.005
    assert result["cpu_init_before_mib"] > 200 and result["cpu_init_after_mib"] == 0
    dist.destroy_process_group()
    return result


@app.local_entrypoint()
def main(
    output: str = ".modal-dojo/new_models/Kimi_K3_H200/fp8-checkpoint-result.json",
):
    result = check.remote()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
