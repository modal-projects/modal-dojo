"""Train a model on an environment with the model's preset recipe."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from modal_dojo.common.budget import Budget
from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.errors import DojoConfigError
from modal_dojo.common.fallback import Fallback, launch_with_fallback
from modal_dojo.common.train import TrainConfig

if TYPE_CHECKING:
    from modal_dojo.common.environment import Environment
    from modal_dojo.common.models import ModelConfig
    from modal_dojo.common.train import TrainingRun
    from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe
    from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe


def _preset_recipe(model: ModelConfig) -> SlimeRecipe | MilesRecipe:
    from modal_dojo.train_recipes.miles_recipe.recipe import MilesRecipe
    from modal_dojo.train_recipes.slime_recipe.recipe import SlimeRecipe

    try:
        return SlimeRecipe.get_base_recipe(model)
    except DojoConfigError:
        pass
    recipe = MilesRecipe.get_base_recipe(model)
    if recipe is None:
        raise DojoConfigError(
            f"No recipe preset for {model.model_name!r}. Use TrainConfig with an "
            "explicit recipe for this model."
        )
    return recipe


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

    With ``fallback_recipes``, ``launch()`` tries each recipe in turn, moving
    on when one has not started training within the threshold (see
    ``Fallback``). Proof of concept.

    Attributes:
        model: The model to train; it needs a slime or Miles preset.
        environment: The dataset, reward, and optional rollout.
        budget: When to stop.
        recipe_overrides: Recipe fields applied on top of the preset.
        fallback_recipes: Recipes to try in order, instead of the preset.
        train_config: The built ``TrainConfig`` (the first, with fallbacks).
        train_configs: Every ``TrainConfig`` ``launch()`` may try, in order.
    """

    def __init__(
        self,
        model: ModelConfig,
        environment: Environment,
        budget: Budget,
        recipe_overrides: dict[str, Any] | None = None,
        fallback_recipes: Fallback | None = None,
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
        self.fallback_recipes = fallback_recipes
        bases = (
            list(fallback_recipes.recipes)
            if fallback_recipes is not None
            else [_preset_recipe(model)]
        )
        self.train_configs = [
            TrainConfig(
                dataset=environment.dataset,
                model=model,
                recipe=self._build_recipe(base),
            )
            for base in bases
        ]

    @property
    def train_config(self) -> TrainConfig:
        """The first ``TrainConfig``: the preset's, or the primary fallback recipe's."""
        return self.train_configs[0]

    def _build_recipe(
        self, preset: SlimeRecipe | MilesRecipe
    ) -> SlimeRecipe | MilesRecipe:
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
        """Launch, then wait for training to finish; see ``TrainConfig.train``."""
        if self.fallback_recipes is None:
            return self.train_config.train(show_output=show_output)
        return self.launch(show_output=show_output).result(stop_app_on_success=True)

    def launch(self, *, show_output: bool = True) -> TrainingRun:
        """Start training detached, falling back through recipes if given."""
        if self.fallback_recipes is None:
            return self.train_config.launch(show_output=show_output)
        return launch_with_fallback(
            self.train_configs,
            self.fallback_recipes.threshold,
            launch=lambda config: config.launch(show_output=show_output),
        )
