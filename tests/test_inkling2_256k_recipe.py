from pathlib import Path

import pytest

from modal_dojo import Inkling_2_Small, Inkling_2_Small_LoRA_256k_Recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_inkling2_long_context as patcher,
    patch_glm_5_3_flash_moe_seq_chunk as moe_patcher,
)
from modal_dojo.train_recipes.miles_recipe import (
    Inkling_2_Small_LoRA_Recipe,
    MilesRecipe,
)


def test_256k_cli_and_base_selection():
    recipe = Inkling_2_Small_LoRA_256k_Recipe()
    cli = recipe.cli_args()
    for flag, value in {
        "--max-seq-len": "262144",
        "--rollout-max-prompt-len": "16384",
        "--rollout-max-response-len": "245760",
        "--eval-max-response-len": "245760",
        "--seq-length": "262144",
        "--sglang-context-length": "262144",
        "--micro-batch-size": "1",
        "--tito-model": "inkling",
        "--use-session-server": "v1",
    }.items():
        assert cli[cli.index(flag) + 1] == value
    assert "--rm-type" not in cli
    assert "--use-dynamic-batch-size" not in cli
    assert "--rollout-max-context-len" not in cli
    assert (
        recipe.global_batch_size
        == recipe.rollout_batch_size * recipe.n_samples_per_prompt
    )
    assert recipe.custom_model_provider_path.endswith(
        "inkling2_long_context.model_provider"
    )
    assert recipe.docker_image == Inkling_2_Small_LoRA_Recipe().docker_image
    assert (
        type(MilesRecipe.get_base_recipe(Inkling_2_Small()))
        is Inkling_2_Small_LoRA_Recipe
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_max_response_len": 262144},
        {"sglang_context_length": 8192},
        {"context_parallel_size": 2},
        {"sequence_parallel": True},
        {"use_dynamic_batch_size": True},
        {"micro_batch_size": 2},
        {"extra_config": {"recompute_loss_function": True}},
        {"extra_config": {"context_parallel_size": 2}},
        {"extra_config": {"rm_type": "math"}},
    ],
)
def test_256k_rejects_unsupported_effective_values(overrides):
    with pytest.raises(ValueError):
        Inkling_2_Small_LoRA_256k_Recipe(**overrides)


def test_required_memory_patches_survive_custom_image_commands():
    required = Inkling_2_Small_LoRA_256k_Recipe().image_run_commands
    customized = Inkling_2_Small_LoRA_256k_Recipe(image_run_commands=["echo custom"])
    assert customized.image_run_commands == [*required, "echo custom"]


def test_swe_proof_preserves_task_metadata_and_reporting_hooks():
    from configs.inkling_2_small_swe_256k import build_config

    async def reward(args, sample):
        return 1.0

    row = {
        "prompt": [{"role": "user", "content": "Fix the issue"}],
        "label": "repo__issue-1",
        "metadata": {"instance_id": "repo__issue-1"},
    }
    config = build_config([row], reward)
    assert list(config.dataset.rows()) == [row]
    assert config.recipe.custom_rm_function is reward
    assert config.recipe.save is None
    args = config.recipe.cli_args()
    assert "--custom-megatron-before-train-step-hook-path" in args
    assert "--custom-rollout-log-function-path" in args
    with pytest.raises(ValueError, match="ten smoke"):
        build_config([row], reward, steps=500)


@pytest.mark.parametrize(
    "target",
    ["layers.py", "logit_processors.py", "transformer_block.py", "moe_layer.py"],
)
def test_pinned_inkling_patch_snapshots(target):
    root = Path(__file__).parent / "testdata/miles/inkling2_long_context"
    original = (root / f"{target}.input").read_text()
    patch = (
        moe_patcher.patch_source
        if target == "moe_layer.py"
        else lambda s: patcher.patch_source(s, target)
    )
    actual = patch(original)
    assert actual == (root / f"{target}.output").read_text()
    compile(actual, target, "exec")
    assert patch(actual) == actual
    with pytest.raises(ValueError):
        patch("# source changed\n")
