from modal_dojo import (
    Qwen3_6_35B,
    Qwen3_6_35B_Recipe,
)
from modal_dojo.frameworks.slime.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


def test_qwen3_6_35b_uses_async_disagg_eight_b300_tp2_ep2() -> None:
    model = Qwen3_6_35B()
    recipe = Qwen3_6_35B_Recipe()

    assert isinstance(SlimeRecipe.get_base_recipe(model), Qwen3_6_35B_Recipe)
    assert recipe.gpu_type == "B300"
    assert recipe.colocate is False
    assert recipe.train_async is True
    assert recipe.actor_num_nodes == 1
    assert recipe.actor_num_gpus_per_node == 4
    assert recipe.gpu_allocation.actor_gpus == 4
    assert recipe.gpu_allocation.rollout_gpus == 4
    assert recipe.gpu_allocation.total_gpus == 8
    assert recipe.gpu_allocation.gpus_per_node == 8
    assert recipe.gpu_allocation.total_nodes == 1
    assert recipe.global_batch_size == (
        recipe.rollout_batch_size * recipe.n_samples_per_prompt
    )
    assert recipe.ref_load == "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp1"

    cli_args = recipe.cli_args(model=model)
    for flag, value in (
        ("--tensor-model-parallel-size", "2"),
        ("--expert-model-parallel-size", "2"),
        ("--max-tokens-per-gpu", "16384"),
    ):
        assert cli_args[cli_args.index(flag) + 1] == value
    assert "--sequence-parallel" in cli_args
    assert "--use-distributed-optimizer" in cli_args
    dispatcher = cli_args.index("--moe-token-dispatcher-type")
    assert cli_args[dispatcher + 1] == "alltoall"
    assert "--moe-enable-deepep" not in cli_args
    ep = cli_args.index("--sglang-ep-size")
    assert cli_args[ep + 1] == "1"
    attn = cli_args.index("--sglang-attention-backend")
    assert cli_args[attn + 1] == "triton"
    moe = cli_args.index("--sglang-moe-runner-backend")
    assert cli_args[moe + 1] == "flashinfer_trtllm_routed"

    nodes, processes, conversion_args = get_checkpoint_conversion_policy(
        recipe, model=model
    )
    assert (nodes, processes) == (1, 2)
    assert "--tensor-model-parallel-size 2" in conversion_args
    assert "--pipeline-model-parallel-size 1" in conversion_args
