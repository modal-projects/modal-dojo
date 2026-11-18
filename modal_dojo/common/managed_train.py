"""Train a model on an environment with the model's preset recipe."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common import models
from modal_dojo.common.budget import Budget
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.models import ModelConfig
from modal_dojo.common.train import TrainConfig
from modal_dojo.train_recipes import miles_recipe, slime_recipe
from modal_dojo.train_recipes.miles_recipe import MilesRecipe
from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

if TYPE_CHECKING:
    from modal_dojo.common.environment import Environment
    from modal_dojo.common.train import TrainingRun


# The preset each model trains with. Lookup walks the model's MRO, so a
# subclass (e.g. a LoRA variant) maps to its own entry before its parent's.
PRESET_RECIPES: dict[type[ModelConfig], type[SlimeRecipe | MilesRecipe]] = {
    # slime
    models.GLM_4_7: slime_recipe.GLM_4_7_Recipe,
    models.Qwen3_0_6B: slime_recipe.Qwen3_0_6B_Recipe,
    models.Qwen3_1_7B: slime_recipe.Qwen3_1_7B_Recipe,
    models.Qwen3_4B: slime_recipe.Qwen3_4B_Recipe,
    models.Qwen3_8B: slime_recipe.Qwen3_8B_Recipe,
    models.Qwen3_5_0_8B: slime_recipe.Qwen3_5_0_8B_Recipe,
    models.Qwen3_5_2B: slime_recipe.Qwen3_5_2B_Recipe,
    models.Qwen3_5_4B: slime_recipe.Qwen3_5_4B_Recipe,
    models.Qwen3_5_9B: slime_recipe.Qwen3_5_9B_Recipe,
    models.Qwen3_6_27B: slime_recipe.Qwen3_6_27B_Recipe,
    models.Qwen3_6_35B: slime_recipe.Qwen3_6_35B_Recipe,
    models.Qwen3_8_27B: slime_recipe.Qwen3_8_27B_Recipe,
    models.Qwen3_ASR_1_7B: slime_recipe.Qwen3_ASR_1_7B_Recipe,
    models.Qwen3_VL_8B: slime_recipe.Qwen3_VL_8B_Recipe,
    # Miles
    models.Beam_1_0_501B_A23B_LoRA: miles_recipe.Beam_1_0_501B_A23B_LoRA_256k_Recipe,
    models.DeepSeek_V4_1_Flash: miles_recipe.DeepSeek_V4_1_Flash_Recipe,
    models.DeepSeek_V4_1_Flash_LoRA: miles_recipe.DeepSeek_V4_1_Flash_LoRA_Recipe,
    models.Gemma4_26B_A4B: miles_recipe.Gemma4_26B_A4B_Recipe,
    models.GLM_5_3_Flash_LoRA: miles_recipe.GLM_5_3_Flash_LoRA_Recipe,
    models.Inkling_Small: miles_recipe.Inkling_Small_Recipe,
    models.Inkling_Small_LoRA: miles_recipe.Inkling_Small_LoRA_Recipe,
    models.Kimi_K3: miles_recipe.Kimi_K3_LoRA_Recipe,
    models.Moonlight_16B_A3B_Instruct: miles_recipe.Moonlight_16B_A3B_Recipe,
    models.Muse_Spark_Open: miles_recipe.Muse_Spark_Open_64k_Recipe,
}


def _preset_recipe(model: ModelConfig) -> SlimeRecipe | MilesRecipe:
    for cls in type(model).__mro__:
        if cls in PRESET_RECIPES:
            return PRESET_RECIPES[cls]()
    raise DojoConfigError(
        f"No recipe preset for {type(model).__name__} ({model.model_name!r}). "
        "Use TrainConfig with an explicit recipe for this model."
    )


def _context_fields(preset: SlimeRecipe | MilesRecipe, context: int) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rollout_max_response_len": context - 1,
        "eval_max_response_len": context - 1,
    }
    if isinstance(preset.max_tokens_per_gpu, int):
        fields["max_tokens_per_gpu"] = max(preset.max_tokens_per_gpu, context)
    if hasattr(preset, "max_seq_len"):
        fields["max_seq_len"] = context
    else:
        fields["extra_config"] = {
            **(preset.extra_config or {}),
            "rollout_max_context_len": context,
        }
    return fields


# TODO: add to the API reference once the API shape is final.
@exclude_from_api_reference
class ManagedTrainConfig:
    """Train a model on an environment with its preset recipe until ``budget`` runs out.

    ```python
    run = ManagedTrainConfig(
        Qwen3_5_4B(),
        Environment(
            dataset=my_dataset, reward_fn=my_reward, max_context_len=32_768
        ),
        Budget(max_cost_usd=50, max_hours=6),
    ).train()
    ```

    Attributes:
        model: The model to train; it needs a slime or Miles preset.
        environment: The dataset, reward, and optional rollout.
        budget: When to stop.
        recipe_overrides: Recipe fields applied on top of the preset.
        train_config: The built ``TrainConfig``.
    """

    def __init__(
        self,
        model: ModelConfig,
        environment: Environment,
        budget: Budget,
        recipe_overrides: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(budget, Budget):
            raise DojoConfigError(
                f"budget must be a Budget, got {type(budget).__name__}; for a "
                "fixed number of steps use Budget(max_rollouts=...)"
            )
        self.model = model
        self.environment = environment
        self.budget = budget
        self.recipe_overrides = dict(recipe_overrides or {})
        self.train_config = TrainConfig(
            dataset=environment.dataset,
            model=model,
            recipe=self._build_recipe(),
        )

    def _build_recipe(self) -> SlimeRecipe | MilesRecipe:
        preset = _preset_recipe(self.model)
        env = self.environment
        overrides = self.recipe_overrides
        unknown = set(overrides) - {f.name for f in dataclasses.fields(preset)}
        governed = {"num_rollout", "budget"} & set(overrides)
        if governed:
            raise DojoConfigError(
                f"Set {sorted(governed)} through budget, not recipe_overrides"
            )
        if unknown:
            raise DojoConfigError(
                f"Unknown {type(preset).__name__} field(s) in recipe_overrides: "
                f"{sorted(unknown)}"
            )

        derived: dict[str, Any] = {
            "num_rollout": self.budget.num_rollout,
            "budget": self.budget,
            "custom_rm_function": env.reward_fn,
        }
        if env.generate_fn is not None:
            derived["custom_generate_function"] = env.generate_fn
        if env.max_context_len is not None:
            derived.update(_context_fields(preset, env.max_context_len))
        num_tasks = len(list(env.dataset.rows()))
        if num_tasks < preset.rollout_batch_size:
            samples = overrides.get("n_samples_per_prompt", preset.n_samples_per_prompt)
            derived["rollout_batch_size"] = num_tasks
            derived["global_batch_size"] = num_tasks * samples
        return dataclasses.replace(preset, **{**derived, **overrides})

    @property
    def recipe(self) -> SlimeRecipe | MilesRecipe:
        return self.train_config.recipe

    def train(self, *, show_output: bool = True) -> TrainingRun:
        return self.train_config.train(show_output=show_output)

    def launch(self, *, show_output: bool = True) -> TrainingRun:
        return self.train_config.launch(show_output=show_output)
