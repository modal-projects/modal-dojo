"""Guard the Beam runtime/parallelism contract before expensive cluster launches."""

import pytest

from modal_dojo import (
    Beam_1_0_501B_A23B,
    Beam_1_0_501B_A23B_LoRA,
    Beam_1_0_501B_A23B_Recipe,
    Beam_1_0_501B_A23B_LoRA_Recipe,
)
from modal_dojo.frameworks.miles.modal_helpers.utils import model_args_command
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from scripts.validation_backends.miles import build_miles_validation


@pytest.mark.parametrize(
    "model,recipe_type,total",
    [
        (Beam_1_0_501B_A23B, Beam_1_0_501B_A23B_Recipe, 96),
        (Beam_1_0_501B_A23B_LoRA, Beam_1_0_501B_A23B_LoRA_Recipe, 32),
    ],
)
def test_beam_launch_contract(model, recipe_type, total):
    recipe, dataset = build_miles_validation(model(), 1)
    assert isinstance(recipe, recipe_type)
    assert isinstance(MilesRecipe.get_base_recipe(model()), recipe_type)
    assert recipe.gpu_allocation.total_gpus == total
    assert recipe.async_mode and not recipe.colocate
    assert "beam" in model_args_command(recipe, "/root/miles")
    assert recipe.custom_rm_function is not None
    args = recipe.cli_args(model=model(), dataset=dataset)
    assert "--use-dynamic-batch-size" not in args
    assert "--accumulate-allreduce-grads-in-fp32" not in args
    assert "--grad-reduce-in-bf16" in args
    assert "--custom-megatron-before-train-step-hook-path" in args
    assert "--miles-git-ref" not in args
    assert "--sglang-git-ref" not in args


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_dynamic_batch_size": True},
        {"qkv_format": "thd"},
        {"attention_backend": "flash"},
        {"colocate": True},
    ],
)
def test_reject_unsupported_attention_and_async_layout(overrides):
    with pytest.raises(ValueError):
        Beam_1_0_501B_A23B_Recipe(**overrides)


def test_lora_requires_unsharded_expert_adapters():
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        Beam_1_0_501B_A23B_LoRA_Recipe(expert_tensor_parallel_size=2)
    with pytest.raises(ValueError, match="positive lora_rank"):
        Beam_1_0_501B_A23B_LoRA_Recipe(lora_rank=0)
    args = Beam_1_0_501B_A23B_LoRA_Recipe().cli_args()
    assert "--stream-optimizer-state-to-disk" not in args
    assert "--target-modules" not in args
