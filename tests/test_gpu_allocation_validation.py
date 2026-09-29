import warnings
from types import SimpleNamespace

import pytest

from modal_dojo.common.errors import GpuAllocationError
from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe
from modal_dojo.train_recipes.gpu_allocation import (
    resolve_gpu_allocation,
    validate_megatron_actor_parallelism,
    validate_num_experts_divisible_by_expert_parallel_size,
)
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


_SLIME_KW = dict(
    gpu_type="H100",
    colocate=False,
    actor_num_gpus_per_node=8,
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


def test_multi_node_requires_full_node_gpus() -> None:
    with pytest.raises(ValueError, match="8 GPUs per node"):
        SlimeRecipe(
            **{
                **_SLIME_KW,
                "gpu_type": "B300",
                "actor_num_nodes": 2,
                "actor_num_gpus_per_node": 4,
                "colocate": True,
            }
        )


def test_single_node_rejects_gpus_over_container_max() -> None:
    with pytest.raises(ValueError, match="exceeds the 4 GPU container limit"):
        SlimeRecipe(
            **{
                **_SLIME_KW,
                "gpu_type": "A10",
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 8,
                "colocate": True,
            }
        )


@pytest.mark.parametrize("gpu_type", ["A10", "A10!"])
def test_a10_one_plus_three_packs_onto_one_node(gpu_type: str) -> None:
    config = SimpleNamespace(
        gpu_type=gpu_type,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        use_critic=False,
        rollout_num_gpus=3,
    )

    allocation = resolve_gpu_allocation(config, warn=False)
    assert allocation.actor_gpus == 1
    assert allocation.rollout_gpus == 3
    assert allocation.gpus_per_node == 4
    assert allocation.total_gpus == 4
    assert allocation.total_nodes == 1


def test_disagg_three_plus_one_packs_onto_one_node() -> None:
    config = SimpleNamespace(
        gpu_type="H100",
        actor_num_nodes=1,
        actor_num_gpus_per_node=3,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        use_critic=False,
        rollout_num_gpus=1,
    )

    allocation = resolve_gpu_allocation(config, warn=False)
    assert allocation.actor_gpus == 3
    assert allocation.rollout_gpus == 1
    assert allocation.gpus_per_node == 4
    assert allocation.total_gpus == 4
    assert allocation.total_nodes == 1


def test_disagg_one_plus_one_packs_onto_one_node() -> None:
    config = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        rollout_num_gpus_per_engine=1,
        colocate=False,
        use_critic=False,
        rollout_num_gpus=1,
    )

    allocation = resolve_gpu_allocation(config, warn=False)
    assert allocation.actor_gpus == 1
    assert allocation.rollout_gpus == 1
    assert allocation.critic_gpus == 0
    assert allocation.gpus_per_node == 2
    assert allocation.total_gpus == 2
    assert allocation.total_nodes == 1


def test_miles_uses_same_gpu_allocation_math() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = MilesRecipe(
            colocate=False,
            actor_num_gpus_per_node=8,
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
        MilesRecipe(actor_num_gpus_per_node=8.5)


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


def test_num_experts_validation_skips_unset_expert_parallel_size() -> None:
    # An unset EP falls back to the framework default of 1, which always divides.
    config = SimpleNamespace(expert_model_parallel_size=None)
    model = SimpleNamespace(architecture=SimpleNamespace(num_experts=160))

    validate_num_experts_divisible_by_expert_parallel_size(config, model)


def _moe_model(num_experts: int) -> SimpleNamespace:
    return SimpleNamespace(architecture=SimpleNamespace(num_experts=num_experts))


def test_miles_validates_num_experts_against_expert_parallel_size() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recipe = MilesRecipe(expert_model_parallel_size=4)

    with pytest.raises(
        GpuAllocationError,
        match=r"num_experts=10.*expert_model_parallel_size=4",
    ):
        recipe.validate_model_parallelism(_moe_model(10))

    recipe.validate_model_parallelism(_moe_model(8))


def test_miles_default_expert_parallel_size_divides_any_expert_count() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recipe = MilesRecipe()

    recipe.validate_model_parallelism(_moe_model(160))
