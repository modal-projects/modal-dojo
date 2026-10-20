"""Compatibility and full-context stress checks for the pinned V4.1 LoRA stack."""

import ast
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo import (
    DeepSeek_V4_1_Flash_LoRA,
    DeepSeek_V4_1_Flash_LoRA_Recipe,
    MilesRecipe,
)
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_deepseek_v41_vision_topk_capture as capture,
    patch_substep_timing as timing,
)
from modal_dojo.frameworks.miles.modal_helpers.convert_hf_to_torch_dist import (
    keep_iteration_directory,
)
from scripts.validate_deepseek_v41_lora import fill_context

DATA = Path(__file__).parent / "testdata"


def test_conversion_keeps_all_nodes_in_iteration_directory(request):
    source = (DATA / "miles/convert_hf_to_torch_dist_65e785f.py.input").read_text()
    actual = keep_iteration_directory(source)
    golden = DATA / "miles/convert_hf_to_torch_dist_65e785f.py.output"
    if request.config.getoption("--rewrite"):
        golden.write_text(actual)
    assert actual == golden.read_text()
    compile(actual, "convert_hf_to_torch_dist.py", "exec")
    assert "publish_as_release(args.save)" not in actual
    assert 'f.write("1")' in actual


@pytest.mark.parametrize(
    "rename", ["publish_as_release(args.save)", "shutil.move(source_dir, target_dir)"]
)
def test_conversion_does_not_rename_either_upstream_layout(tmp_path, rename):
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    source = f'{rename}\nwith open(tracker, "w") as f:\n    f.write("release")\n'
    # No rename function exists in the namespace: calling one fails this test.
    exec(keep_iteration_directory(source), {"tracker": tracker})
    assert tracker.read_text() == "1"


def test_conversion_rejects_unknown_publication_code():
    with pytest.raises(RuntimeError, match="refusing a multi-node rename"):
        keep_iteration_directory('new_publication_api()\nf.write("release")')


def test_model_resolves_native_lora_recipe():
    recipe = MilesRecipe.get_base_recipe(DeepSeek_V4_1_Flash_LoRA())
    assert isinstance(recipe, DeepSeek_V4_1_Flash_LoRA_Recipe)
    assert recipe.lora_rank == recipe.sglang_max_lora_rank == 32
    assert recipe.target_modules.split(",") == [
        f"layers.*.attn.{name}" for name in ("wq_a", "wkv", "wq_b", "wo_b")
    ]
    assert recipe.rollout_max_context_len == recipe.sglang_context_length == 32768
    assert recipe.gpu_allocation.total_gpus == 32


@pytest.mark.parametrize(
    "evaluation,prompt_length", [(False, 3), (False, 524), (True, 3)]
)
def test_stress_generation_fills_remaining_context_only_in_training(
    monkeypatch, evaluation, prompt_length
):
    params = {"max_new_tokens": 4096, "temperature": 1.0}
    sample = SimpleNamespace(
        prompt="prompt",
        tokens=[],
        status="done",
        Status=SimpleNamespace(ABORTED="aborted"),
    )

    async def generate(args, sample, sampling_params, *, evaluation):
        if evaluation:
            assert sampling_params == params
            sample.tokens = [0] * 10
        else:
            assert sampling_params["max_new_tokens"] == 32766 - prompt_length
            assert sampling_params["ignore_eos"] is True
            assert sampling_params["stop"] == sampling_params["stop_token_ids"] == []
            sample.tokens = [0] * 32766
        return sample

    monkeypatch.setitem(
        sys.modules,
        "miles.rollout.sglang_rollout",
        SimpleNamespace(
            GenerateState=lambda args: SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda *a, **kw: [0] * prompt_length)
            ),
            generate=generate,
        ),
    )
    asyncio.run(
        fill_context(
            SimpleNamespace(rollout_max_context_len=32768),
            sample,
            params,
            evaluation=evaluation,
        )
    )
    assert params == {"max_new_tokens": 4096, "temperature": 1.0}


def test_stress_generation_rejects_short_completion(monkeypatch):
    sample = SimpleNamespace(
        prompt="prompt",
        tokens=[0] * 100,
        status="done",
        Status=SimpleNamespace(ABORTED="aborted"),
    )

    async def generate(*args, **kwargs):
        return sample

    monkeypatch.setitem(
        sys.modules,
        "miles.rollout.sglang_rollout",
        SimpleNamespace(
            GenerateState=lambda args: SimpleNamespace(
                tokenizer=SimpleNamespace(encode=lambda *a, **kw: [0])
            ),
            generate=generate,
        ),
    )
    with pytest.raises(ValueError, match="expected 32766"):
        asyncio.run(
            fill_context(SimpleNamespace(rollout_max_context_len=32768), sample, {})
        )


def test_capture_patch_pinned_source_snapshot(request):
    source = (DATA / "sglang/vl_routing_34033fe.py.input").read_text()
    actual = capture.patch_source(source)
    golden = DATA / "sglang/vl_routing_34033fe.py.output"
    if request.config.getoption("--rewrite"):
        golden.write_text(actual)
    assert actual == golden.read_text()
    assert capture.patch_source(actual) == actual
    compile(actual, "vl_routing.py", "exec")


@pytest.mark.parametrize(
    "mega,padded", [(False, False), (False, True), (True, False), (True, True)]
)
def test_expert_capture_once_after_padding(monkeypatch, mega, padded):
    source = capture.patch_source(
        (DATA / "sglang/vl_routing_34033fe.py.input").read_text()
    )
    fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_finish_vision_topk"
    )
    events = []

    def postprocess(indices, weights, *args, **kwargs):
        events.append("capture")
        return indices, weights, None

    ns = {
        "get_moe_a2a_backend": lambda: SimpleNamespace(is_megamoe=lambda: mega),
        "_post_process_topk_ids": postprocess,
        "_scale_fused_shared_weights": lambda weights, *a: weights,
        "_mask_topk_ids_padded_region": lambda *a: events.append("mask"),
        "_zero_topk_weights_padded_region": lambda *a: events.append("zero"),
        "capture_routed_experts_if_allowed": lambda *a: events.append("capture"),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "vl_routing.py", "exec"), ns)
    config = SimpleNamespace(
        num_fused_shared_experts=0, fused_shared_experts_scaling_factor=None
    )
    moe = SimpleNamespace(topk=SimpleNamespace(topk_config=config), layer_id=3)
    ns["_finish_vision_topk"](moe, None, [], [], 1, None, padded_rows_masked=padded)
    assert events.count("capture") == 1
    if not mega and not padded:
        assert events == ["mask", "zero", "capture"]


def test_weight_sync_timing_new_checksum_api(tmp_path, request):
    relative = "miles/ray/placement_group.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text((DATA / "miles/placement_group_65e785f.py.input").read_text())
    timing.patch_executor_package(tmp_path)
    actual = target.read_text()
    assert timing.PREAMBLE_MARKER in actual
    for phase in (
        "wait_for_inference_engines",
        "initial_weight_sync",
        "finalize_weight_sync",
        "check_weight_sync",
    ):
        assert phase in actual
    golden = DATA / "miles/placement_group_65e785f.py.timing.output"
    if request.config.getoption("--rewrite"):
        golden.write_text(actual)
    assert actual == golden.read_text()
    compile(actual, relative, "exec")
