import pytest

from modal_dojo import (
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    Qwen3_8_27B,
    Qwen3_8_27B_Recipe,
)
from modal_dojo.frameworks.slime.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe


@pytest.mark.parametrize(
    ("model_cls", "recipe_cls", "ref_load"),
    [
        (Qwen3_6_27B, Qwen3_6_27B_Recipe, "/checkpoints/Qwen3.6-27B_torch_dist_tp1pp1"),
        (Qwen3_8_27B, Qwen3_8_27B_Recipe, "/checkpoints/Qwen3.8-27B_torch_dist_tp1pp1"),
    ],
    ids=["qwen3.6-27b", "qwen3.8-27b"],
)
def test_qwen3_27b_adapts_current_slime_qwen3_5_recipe(
    model_cls, recipe_cls, ref_load
) -> None:
    model = model_cls()
    recipe = recipe_cls()

    assert isinstance(SlimeRecipe.get_base_recipe(model), recipe_cls)
    assert recipe.slime_model_script == "scripts/models/qwen3.5-27B.sh"
    assert recipe.hf_checkpoint == model.model_name
    assert recipe.ref_load == ref_load
    assert recipe.gpu_type == "B300"
    assert recipe.attention_backend == "unfused"
    assert recipe.calculate_per_token_loss is True
    assert recipe.max_tokens_per_gpu == 8192
    assert recipe.memory == (128, 2_097_152)

    fields = recipe._fields(model=model)
    assert fields["hf_checkpoint"] == model.model_name
    assert "num_layers" not in fields
    assert "spec" not in fields
    cli_args = recipe.cli_args(model=model)
    prefill_backend = cli_args.index("--sglang-cuda-graph-backend-prefill")
    assert cli_args[prefill_backend + 1] == "disabled"

    nodes, processes, conversion_args = get_checkpoint_conversion_policy(
        recipe, model=model
    )
    assert (nodes, processes) == (1, 1)
    assert all("tensor-model-parallel-size" not in arg for arg in conversion_args)
    assert all("pipeline-model-parallel-size" not in arg for arg in conversion_args)
