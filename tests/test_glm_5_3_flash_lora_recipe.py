"""Bridge LoRA must select its own preset and preserve FFT behavior."""

from pathlib import Path

import pytest

from modal_training_gym import (
    GLM_5_3_Flash,
    GLM_5_3_Flash_LoRA,
    GLM_5_3_Flash_LoRA_Recipe,
    GLM_5_3_Flash_Recipe,
)
from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_lora_timing as timing,
)
from modal_training_gym.frameworks.miles.modal_helpers.utils import build_train_cmd
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

SNAPSHOTS = Path(__file__).parent / "testdata/miles/glm_5_3_flash_lora"


def test_lora_preset_selection_and_bridge_command():
    model = GLM_5_3_Flash_LoRA()
    recipe = MilesRecipe.get_base_recipe(model)
    assert isinstance(recipe, GLM_5_3_Flash_LoRA_Recipe)
    assert recipe.gpu_allocation.total_gpus == 24
    assert recipe.global_batch_size % (24 // recipe.tensor_model_parallel_size) == 0
    cmd = build_train_cmd(recipe, "/root/miles", model=model)
    for flag in (
        "--megatron-to-hf-mode bridge",
        "--lora-rank 16",
        "--lora-alpha 32",
        "--expert-model-parallel-size 24",
        "--pipeline-model-parallel-size 1",
        "--lora-base-cpu-backup",
        "--check-lora-weight-equal",
        "--sglang-disable-shared-experts-fusion",
        "--offload-train-target cpu",
    ):
        assert flag in cmd
    assert "--ref-load" not in cmd
    assert "--experts-shared-outer-loras" not in cmd
    assert "--use-dynamic-batch-size" not in cmd
    assert "phase_reporting.before_train_step_hook" in cmd
    assert len(recipe.target_modules.split(",")) == 19
    assert "decoder.layers.*.self_attention.linear_g_b" in recipe.target_modules
    assert "decoder.layers.*.mlp.experts.linear_fc2" in recipe.target_modules
    assert "indexer" not in recipe.target_modules


def test_fft_still_selects_original_raw_recipe():
    recipe = MilesRecipe.get_base_recipe(GLM_5_3_Flash())
    assert isinstance(recipe, GLM_5_3_Flash_Recipe)
    assert recipe.megatron_to_hf_mode == "raw"
    assert recipe.gpu_allocation.total_gpus == 32
    assert recipe.lora_rank is None
    assert recipe.sglang_git_ref is None


def test_rank_must_match_serving_capacity():
    with pytest.raises(ValueError, match="must match lora_rank"):
        GLM_5_3_Flash_LoRA_Recipe(lora_rank=32)
    assert (
        GLM_5_3_Flash_LoRA_Recipe(lora_rank=32, sglang_max_lora_rank=32).lora_rank == 32
    )


@pytest.mark.parametrize("name", ["train.py", "actor.py", "model.py"])
def test_pinned_bridge_timing(name):
    source = (SNAPSHOTS / f"{name}.input").read_text()
    actual = timing.patch_source(source, name)
    assert actual == (SNAPSHOTS / f"{name}.output").read_text()
    compile(actual, name, "exec")
    assert timing.patch_source(actual, name) == actual


def test_timing_covers_initial_and_step_sync_and_rejects_drift():
    source = (SNAPSHOTS / "train.py.input").read_text()
    patched = timing.patch_source(source, "train.py")
    for phase in (
        "initial_weight_sync",
        "weight_sync",
        "train_models",
        "generate_rollouts",
    ):
        assert f"_tg_glm_phase('{phase}')" in patched
    with pytest.raises(ValueError, match="actor_model.update_weights"):
        timing.patch_source(
            source.replace("actor_model.update_weights", "actor_model.sync"), "train.py"
        )
