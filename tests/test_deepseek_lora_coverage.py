"""Recipe and pinned-source contracts for the broader native V4.1 adapter."""

import hashlib
import json
from pathlib import Path

import pytest

from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_deepseek_v41_lora_efficiency as patch,
)
from modal_dojo.train_recipes.miles_recipe.deepseek_v41_flash_lora import (
    MILES_LORA_COMMIT,
)

FIXTURES = Path(__file__).parent / "testdata/deepseek_lora_coverage"


def test_coverage_and_training_serving_contract():
    recipe = DeepSeek_V4_1_Flash_LoRA_Recipe()
    assert set(recipe.target_modules.split(",")) == {
        "layers.*.attn.wq_a",
        "layers.*.attn.wkv",
        "layers.*.attn.wq_b",
        "layers.*.attn.wo_a",
        "layers.*.attn.wo_b",
        "layers.*.ffn.shared_experts.w1",
        "layers.*.ffn.shared_experts.w3",
        "layers.*.ffn.shared_experts.w2",
    }
    assert recipe.exclude_modules == "layers.*.ffn.experts.*"
    assert recipe.lora_rank == recipe.sglang_max_lora_rank == 32
    assert recipe.check_lora_weight_equal and recipe.sglang_lora_strict_loading
    assert not recipe.check_weight_update
    assert recipe.qkv_format == "bshd" and recipe.micro_batch_size == 1
    assert not recipe.use_dynamic_batch_size
    assert recipe.actor_num_nodes * recipe.actor_num_gpus_per_node == 32
    assert recipe.rollout_batch_size * recipe.n_samples_per_prompt == 32
    # Existing adapter checkpoints must still be able to select their old layout.
    old_targets = (
        "layers.*.attn.wq_a,layers.*.attn.wkv,layers.*.attn.wq_b,layers.*.attn.wo_b"
    )
    legacy = DeepSeek_V4_1_Flash_LoRA_Recipe(target_modules=old_targets)
    assert legacy.target_modules == old_targets


def test_pinned_efficiency_patch_is_complete_and_idempotent(tmp_path, request):
    manifest = json.loads((FIXTURES / "SOURCE.json").read_text())
    assert manifest["revision"] == MILES_LORA_COMMIT
    for name, spec in manifest["files"].items():
        source = (FIXTURES / f"{name}.input").read_bytes()
        assert hashlib.sha256(source).hexdigest() == spec["sha256"]
        target = tmp_path / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    patch.apply(tmp_path)
    outputs = {}
    for name, spec in manifest["files"].items():
        result = (tmp_path / spec["path"]).read_text()
        compile(result, name, "exec")
        golden = FIXTURES / f"{name}.output"
        if request.config.getoption("--rewrite"):
            golden.write_text(result)
        assert result == golden.read_text()
        outputs[name] = result
    patch.apply(tmp_path)
    assert outputs == {
        name: (tmp_path / spec["path"]).read_text()
        for name, spec in manifest["files"].items()
    }


@pytest.mark.parametrize(
    ("function", "path"),
    [
        (patch.patch_initial_lora_offload, "ray/rollout/server_cell.py"),
        (patch.patch_microbatch_padding, "backends/training_utils/data/rollout.py"),
    ],
)
def test_patch_refuses_source_drift(tmp_path, function, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("# changed upstream implementation\n")
    with pytest.raises(ValueError, match="anchor changed"):
        function(tmp_path)
