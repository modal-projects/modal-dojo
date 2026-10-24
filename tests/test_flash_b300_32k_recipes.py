"""Check B300 presets through the same preparation path used by launchers."""

from dataclasses import asdict

import pytest

import modal_dojo
from modal_dojo import (
    DeepSeek_V4_1_Flash_B300_32k_Recipe,
    DeepSeek_V4_1_Flash_LoRA_B300_32k_Recipe,
    DeepSeek_V4_1_Flash_LoRA_Recipe,
    DeepSeek_V4_1_Flash_Recipe,
    GLM_5_3_Flash_LoRA_B300_32k_Recipe,
    GLM_5_3_Flash_LoRA_Recipe,
    HuggingFaceDataset,
    TrainConfig,
)
from modal_dojo.common.launcher_utils import get_checkpoint_conversion_policy
from modal_dojo.train_recipes import miles_recipe
from modal_dojo.train_recipes.gpu_allocation import validate_megatron_actor_parallelism
from modal_dojo.train_recipes.miles_recipe import MilesRecipe


PAIRS = [
    (DeepSeek_V4_1_Flash_B300_32k_Recipe, DeepSeek_V4_1_Flash_Recipe, 32, 384),
    (DeepSeek_V4_1_Flash_LoRA_B300_32k_Recipe, DeepSeek_V4_1_Flash_LoRA_Recipe, 8, 384),
    (GLM_5_3_Flash_LoRA_B300_32k_Recipe, GLM_5_3_Flash_LoRA_Recipe, 8, 288),
]


@pytest.mark.parametrize("candidate,baseline,gpus,experts", PAIRS)
def test_prepared_recipe_keeps_context_and_runtime(candidate, baseline, gpus, experts):
    recipe = candidate(num_rollout=1)
    model = recipe.model_config_class()
    config = TrainConfig(
        model=model,
        dataset=HuggingFaceDataset(
            hf_repo="unused/local-preflight",
            input_column="prompt",
            output_column="answer",
            input_format="text",
        ),
        recipe=recipe,
    )
    prepared = config._prepare_recipe()
    validate_megatron_actor_parallelism(prepared)
    fields = prepared._fields(model=model)
    for key in ("seq_length", "rollout_max_context_len", "sglang_context_length"):
        assert fields[key] == 32768
    assert (
        fields["rollout_max_prompt_len"] + fields["rollout_max_response_len"] == 32768
    )
    args = prepared.cli_args(model=model)
    assert args[args.index("--sglang-context-length") + 1] == "32768"
    assert prepared.gpu_type == "B300"
    assert (
        prepared.gpu_allocation.actor_gpus == prepared.gpu_allocation.total_gpus == gpus
    )
    assert prepared.expert_model_parallel_size == gpus
    assert experts % prepared.expert_model_parallel_size == 0
    dp = gpus // prepared.tensor_model_parallel_size
    assert prepared.global_batch_size % (dp * prepared.micro_batch_size) == 0
    assert (
        prepared.rollout_batch_size * prepared.n_samples_per_prompt
        == prepared.global_batch_size
    )
    assert prepared.sglang_max_running_requests == 32
    assert get_checkpoint_conversion_policy(prepared)[:2] == (1, 8)

    # Hardware/context overrides must not silently switch training behavior,
    # upstream revisions, checkpoint layout, hooks or model-specific patches.
    before, after = asdict(baseline()), asdict(prepared)
    allowed = {
        "gpu_type",
        "memory",
        "environment",
        "rollout_max_prompt_len",
        "rollout_max_response_len",
        "rollout_max_context_len",
        "seq_length",
        "max_tokens_per_gpu",
        "sglang_context_length",
        "actor_num_nodes",
        "expert_model_parallel_size",
        "conversion_expert_model_parallel_size",
        "sglang_max_running_requests",
    }
    changed = {key for key in before if before[key] != after[key]}
    assert changed <= allowed
    assert after["environment"]["NCCL_MNNVL_ENABLE"] == "0"
    assert prepared.model_config_class is baseline.model_config_class


@pytest.mark.parametrize("candidate,baseline,gpus,experts", PAIRS)
def test_public_registration_and_default_selection(candidate, baseline, gpus, experts):
    assert getattr(modal_dojo, candidate.__name__) is candidate
    assert candidate.__name__ in modal_dojo.__all__
    assert getattr(miles_recipe, candidate.__name__) is candidate
    assert candidate.__name__ in miles_recipe.__all__
    default = MilesRecipe.get_base_recipe(candidate.model_config_class())
    assert type(default) is baseline
    assert default.gpu_type == "H200"
