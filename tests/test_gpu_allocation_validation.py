import warnings
from types import SimpleNamespace

import pytest

from modal_training_gym.common.errors import GpuAllocationError
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesConfig
from modal_training_gym.train_recipes.gpu_allocation import (
    resolve_gpu_allocation,
    validate_megatron_actor_parallelism,
    validate_num_experts_divisible_by_expert_parallel_size,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


_SLIME_KW = dict(
    gpu_type="H100",
    colocate=False,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
    rollout_num_gpus_per_engine=4,
    num_rollout=1,
    rollout_batch_size=16,
    rollout_max_response_len=4096,
    rollout_temperature=1.0,
    save_interval=10,
)


def test_slime_resolves_non_colocated_gpu_allocation() -> None:
    recipe = SlimeRecipe(**_SLIME_KW, rollout_num_gpus=8)

    allocation = recipe.gpu_allocation
    assert allocation.actor_gpus == 8
    assert allocation.rollout_gpus == 8
    assert allocation.total_gpus == 16
    assert allocation.total_nodes == 2
    assert allocation.rollout_engines == 2
    assert recipe.total_nodes == 2


def test_two_trainer_six_rollout_gpus_share_one_physical_node() -> None:
    recipe = SlimeRecipe(
        **{**_SLIME_KW, "rollout_num_gpus_per_engine": 1},
        actor_num_gpus_per_node=2,
        physical_gpus_per_node=8,
        rollout_num_gpus=6,
    )
    allocation = recipe.gpu_allocation
    assert (allocation.actor_gpus, allocation.rollout_gpus) == (2, 6)
    assert (allocation.total_gpus, allocation.total_nodes, allocation.gpus_per_node) == (8, 1, 8)
    assert allocation.rollout_engines == 6
    arguments = recipe.cli_args()
    assert "--physical-gpus-per-node" not in arguments
    assert arguments[arguments.index("--actor-num-gpus-per-node") + 1] == "2"


def test_actor_ranks_cannot_exceed_physical_node_size() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        SlimeRecipe(**_SLIME_KW, physical_gpus_per_node=4)


@pytest.mark.parametrize("value", [0, -1, True, 8.0, "8"])
def test_physical_node_size_is_a_positive_integer(value) -> None:
    config = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        physical_gpus_per_node=value,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        rollout_num_gpus=6,
        use_critic=False,
    )
    with pytest.raises(GpuAllocationError, match="physical_gpus_per_node"):
        resolve_gpu_allocation(config, warn=False)


def test_rollout_gpus_must_divide_rollout_engine_size() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        SlimeRecipe(**_SLIME_KW, rollout_num_gpus=10)


def test_colocated_rollout_gpu_override_warns_when_it_changes_nothing() -> None:
    with pytest.warns(UserWarning, match="colocate=True uses actor GPUs"):
        recipe = SlimeRecipe(
            **{
                **_SLIME_KW,
                "colocate": True,
                "rollout_num_gpus": 16,
            }
        )

    assert recipe.gpu_allocation.total_gpus == 8
    assert recipe.gpu_allocation.rollout_gpus == 0


def test_large_rollout_allocation_warns() -> None:
    with pytest.warns(UserWarning) as caught:
        SlimeRecipe(**_SLIME_KW, rollout_num_gpus=32)

    messages = [str(w.message) for w in caught]
    assert any("more than 2x actor allocation" in message for message in messages)


def test_miles_uses_same_gpu_allocation_math() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = MilesConfig(
            colocate=False,
            rollout_num_gpus=8,
            rollout_num_gpus_per_engine=4,
        )

    allocation = config.gpu_allocation
    assert allocation.actor_gpus == 8
    assert allocation.rollout_gpus == 8
    assert allocation.total_gpus == 16
    assert config.total_nodes == 2


def test_miles_rejects_bad_gpu_count_type() -> None:
    # Pydantic's field validation rejects the fractional GPU count before
    # resolve_gpu_allocation runs; both ValidationError and GpuAllocationError
    # are ValueError subclasses.
    with pytest.raises(ValueError, match="actor_num_gpus_per_node"):
        MilesConfig(actor_num_gpus_per_node=8.5)


@pytest.mark.parametrize("value", [True, 8.0, "8"])
def test_gpu_allocation_rejects_non_int_values(value: object) -> None:
    config = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=value,
        rollout_num_gpus_per_engine=1,
        colocate=True,
        use_critic=False,
        rollout_num_gpus=None,
    )

    with pytest.raises(GpuAllocationError, match="actor_num_gpus_per_node"):
        resolve_gpu_allocation(config, warn=False)


def test_megatron_parallelism_rejects_expert_layout_larger_than_world_size() -> None:
    config = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=4,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=4,
    )

    with pytest.raises(
        GpuAllocationError,
        match=r"world_size=8.*expert_tensor_model_pipeline_parallel size=32",
    ):
        validate_megatron_actor_parallelism(config)


def test_megatron_parallelism_accepts_valid_toolathlon_layout() -> None:
    config = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=4,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
    )

    validate_megatron_actor_parallelism(config)


def test_num_experts_must_divide_expert_parallel_size() -> None:
    config = SimpleNamespace(expert_model_parallel_size=4)
    model = SimpleNamespace(architecture=SimpleNamespace(num_experts=10))

    with pytest.raises(
        GpuAllocationError,
        match=r"num_experts=10.*expert_model_parallel_size=4",
    ):
        validate_num_experts_divisible_by_expert_parallel_size(config, model)


def test_num_experts_validation_skips_dense_models() -> None:
    config = SimpleNamespace(expert_model_parallel_size=4)
    model = SimpleNamespace(architecture=SimpleNamespace(num_experts=0))

    validate_num_experts_divisible_by_expert_parallel_size(config, model)
