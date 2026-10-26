import pytest

from modal_dojo import (
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    Qwen3_8_27B,
    Qwen3_8_27B_Recipe,
)
from modal_dojo.frameworks.miles.modal_helpers.utils import (
    get_checkpoint_conversion_policy,
)
from modal_dojo.train_recipes.miles_recipe import MilesRecipe


@pytest.mark.parametrize(
    ("model_cls", "recipe_cls", "miles_model_name", "ref_load", "conversion_tp"),
    [
        (
            Qwen3_6_27B,
            Qwen3_6_27B_Recipe,
            "qwen3.6-27B",
            "/checkpoints/Qwen3.6-27B_torch_dist_tp1pp1",
            1,
        ),
        (
            Qwen3_8_27B,
            Qwen3_8_27B_Recipe,
            "qwen3.8-27B",
            "/checkpoints/Qwen3.8-27B_torch_dist_tp2pp1",
            2,
        ),
    ],
    ids=["qwen3.6-27b", "qwen3.8-27b"],
)
def test_qwen3_27b_adapts_current_miles_qwen3_5_recipe(
    model_cls, recipe_cls, miles_model_name, ref_load, conversion_tp
) -> None:
    model = model_cls()
    recipe = recipe_cls()

    assert isinstance(MilesRecipe.get_base_recipe(model), recipe_cls)
    assert recipe.miles_model_name == miles_model_name
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
    nodes, processes, conversion_args = get_checkpoint_conversion_policy(
        recipe, model=model
    )
    assert (nodes, processes) == (1, conversion_tp)
    tp_args = [arg for arg in conversion_args if "tensor-model-parallel-size" in arg]
    pp_args = [arg for arg in conversion_args if "pipeline-model-parallel-size" in arg]
    if conversion_tp == 1:
        assert not tp_args and not pp_args
    else:
        assert tp_args == [f"--tensor-model-parallel-size {conversion_tp}"]
        assert pp_args == ["--pipeline-model-parallel-size 1"]
