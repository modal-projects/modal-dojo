"""Regressions for the two fixes still needed by the pinned H200 image."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_deepseek_v41_fp4_dequant_block as fp4,
    patch_deepseek_v41_vision_topk_capture as routing,
)

FIXTURES = Path(__file__).parent / "testdata" / "miles"


@pytest.mark.parametrize(
    ("patcher", "name"),
    [(fp4, "fp8"), (routing, "vl_routing")],
)
def test_patches_match_pinned_sources_and_are_idempotent(patcher, name):
    source = (FIXTURES / f"deepseek_v41_{name}.py.input").read_text()
    patched = patcher.patch_source(source)
    compile(patched, name, "exec")
    assert patcher.patch_source(patched) == patched
    with pytest.raises(ValueError, match="did not match"):
        patcher.patch_source(
            source.replace("    new_weights = []", "    weights = []")
            if name == "fp8"
            else source.replace("if packed_topk is not None:", "if False:")
        )


def test_fp4_both_dequant_paths_use_checkpoint_block_size():
    source = (FIXTURES / "deepseek_v41_fp8.py.input").read_text()
    tree = ast.parse(fp4.patch_source(source))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "cast_e2m1fn_to_e4m3fn"
    ]
    assert len(calls) == 2
    for call in calls:
        assert any(
            k.arg == "fp8_block_size" and ast.unparse(k.value) == "block_n"
            for k in call.keywords
        )
    cast = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    assert "fp8_block_size" in [arg.arg for arg in cast.args.args]
    assert not any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "fp8_block_size" for t in n.targets)
        for n in ast.walk(cast)
    )


@pytest.mark.parametrize("packed", [False, True])
def test_cuda_routing_captures_experts_before_either_return(monkeypatch, packed):
    source = (FIXTURES / "deepseek_v41_vl_routing.py.input").read_text()
    tree = ast.parse(routing.patch_source(source))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "vision_topk"
    )
    gate_module = ModuleType("sglang.kernels.ops.moe.moe_fused_gate")
    indices = object()
    gate_module.moe_fused_gate = lambda *a, **kw: (object(), indices)
    utils_module = ModuleType("sglang.srt.layers.moe.utils")
    utils_module.get_moe_runner_backend = lambda: SimpleNamespace(
        is_flashinfer_mxfp4=lambda: packed
    )
    monkeypatch.setitem(sys.modules, gate_module.__name__, gate_module)
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)
    captures = []
    scope = {
        "is_cuda": lambda: True,
        "torch": SimpleNamespace(empty=lambda *a, **kw: object(), int32="int32"),
        "_RENORMALIZE_SUM_EPSILON": 1e-20,
        "_scale_fused_shared_weights": lambda weights, *a: weights,
        "StandardTopKOutput": lambda *args: args,
        "StandardTopKOutputPacked": lambda *args: args,
        "capture_routed_experts_if_allowed": lambda *args: captures.append(args),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "vision_topk", "exec"),
        scope,
    )
    config = SimpleNamespace(
        num_fused_shared_experts=0,
        top_k=6,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
        fused_shared_experts_scaling_factor=None,
    )
    moe = SimpleNamespace(
        topk=SimpleNamespace(topk_config=config),
        layer_id=3,
        gate=SimpleNamespace(
            e_score_correction_bias=None, e_score_correction_bias_vl=None
        ),
        config=SimpleNamespace(image_token_id=1),
    )
    result = scope["vision_topk"](
        moe, SimpleNamespace(shape=(2, 384), device="cuda"), None
    )
    assert captures == [(config, 3, indices)]
    assert result[1] is indices
    assert len(result) == (4 if packed else 3)
