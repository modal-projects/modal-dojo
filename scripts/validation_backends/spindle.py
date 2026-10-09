"""Validating a model by running base GRPO training on Spindle."""

from __future__ import annotations

from modal_dojo.common.dataset import DatasetConfig, HuggingFaceDataset
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.models import ModelConfig
from modal_dojo.train_recipes.spindle_recipe import SpindleRecipe


def build_spindle_validation(
    model_config: ModelConfig,
    step_count: int,
    *,
    loss_type: str = "policy_loss",
) -> tuple[SpindleRecipe, DatasetConfig]:
    """The model's base Spindle recipe and its validation dataset.

    Spindle recipes are policy-only (tinker-cookbook GRPO), scored with the
    default boxed-answer reward, so they validate against DAPO-Math-17k.
    """
    if loss_type != "policy_loss":
        raise DojoConfigError(
            f"spindle validation supports policy_loss only, got {loss_type!r}"
        )
    recipe = SpindleRecipe.get_base_recipe(model_config)
    recipe.skip_eval_before_train = True
    recipe.rm_type = "deepscaler"
    n_rows = recipe.rollout_batch_size * step_count
    return recipe, HuggingFaceDataset(
        "zhuzilin/dapo-math-17k",
        hf_split=f"train[:{n_rows}]",
        input_column="prompt",
        output_column="label",
        input_format="messages",
        always_download=True,
    )
