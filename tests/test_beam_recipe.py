"""Guard the Beam runtime/parallelism contract before expensive cluster launches."""

import pytest

from modal_dojo import (
    Beam_1_0_501B_A23B_LoRA,
    Beam_1_0_501B_A23B_LoRA_256k_Recipe,
)
from modal_dojo.frameworks.miles.modal_helpers.utils import model_args_command
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from scripts.validation_backends.miles import build_miles_validation


@pytest.mark.parametrize(
    "model,recipe_type,total",
    [
        (Beam_1_0_501B_A23B_LoRA, Beam_1_0_501B_A23B_LoRA_256k_Recipe, 32),
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
        {"attention_backend": "unfused"},
        {"colocate": True},
    ],
)
def test_reject_unsupported_attention_and_async_layout(overrides):
    with pytest.raises(ValueError):
        Beam_1_0_501B_A23B_LoRA_256k_Recipe(**overrides)


def test_lora_requires_unsharded_expert_adapters():
    with pytest.raises(ValueError, match="expert_tensor_parallel_size"):
        Beam_1_0_501B_A23B_LoRA_256k_Recipe(expert_tensor_parallel_size=2)
    with pytest.raises(ValueError, match="positive lora_rank"):
        Beam_1_0_501B_A23B_LoRA_256k_Recipe(lora_rank=0)
    args = Beam_1_0_501B_A23B_LoRA_256k_Recipe().cli_args()
    assert "--stream-optimizer-state-to-disk" not in args
    assert "--target-modules" not in args


def test_256k_beam_preserves_training_workload_and_limits_memory():
    from modal_dojo import Beam_1_0_501B_A23B_LoRA_256k_Recipe

    recipe = Beam_1_0_501B_A23B_LoRA_256k_Recipe()
    assert recipe.gpu_allocation.total_gpus == 32
    assert recipe.rollout_max_response_len == 32768
    assert recipe.rollout_max_prompt_len == 229376
    assert recipe.context_parallel_size == 2
    assert recipe.attention_backend == "flash"
    assert recipe.rollout_max_prompt_len + recipe.rollout_max_response_len == 262144
    assert recipe.seq_length == 262144
    assert recipe.sglang_context_length == 262146
    assert recipe.sglang_max_running_requests == 8
    assert recipe.sglang_cuda_graph_max_bs_decode == 8
    assert recipe.lr == 1e-4
    assert recipe.lr_decay_style == "cosine"
    assert recipe.global_batch_size == 256
    assert recipe.n_samples_per_prompt == 8
    assert recipe.rollout_batch_size == 32
    assert recipe.custom_rm_function is not None
    assert "--cp-comm-type" in recipe.cli_args()


@pytest.mark.parametrize(
    "overrides",
    [
        {"attention_backend": "unfused"},
        {"cp_comm_type": "p2p"},
        {"context_parallel_size": 1},
        {"rollout_max_prompt_len": 262144},
        {"sglang_context_length": 5120},
        {"sglang_context_length": 262144},
        {"seq_length": 5120},
    ],
)
def test_256k_beam_rejects_inconsistent_windows(overrides):
    from modal_dojo import Beam_1_0_501B_A23B_LoRA_256k_Recipe

    with pytest.raises(ValueError):
        Beam_1_0_501B_A23B_LoRA_256k_Recipe(**overrides)
