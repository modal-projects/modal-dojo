"""Route a model, environment, and budget to a preset recipe."""

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


Recipe = SlimeRecipe | MilesRecipe


def _context_fields(preset: Recipe, context: int) -> dict[str, Any]:
    prompt = getattr(preset, "rollout_max_prompt_len", None)
    if isinstance(prompt, int):
        # Keep the preset's prompt/response split, scaled to the new context.
        prompt = prompt * context // (prompt + preset.rollout_max_response_len)
        fields: dict[str, Any] = {"rollout_max_prompt_len": prompt}
    else:
        prompt = 1
        fields = {}
    fields["rollout_max_response_len"] = context - prompt
    fields["eval_max_response_len"] = context - prompt
    if isinstance(preset.max_tokens_per_gpu, int):
        fields["max_tokens_per_gpu"] = max(preset.max_tokens_per_gpu, context)
    for name in ("max_seq_len", "rollout_max_context_len"):
        if hasattr(preset, name):
            fields[name] = context
            break
    else:
        fields["extra_config"] = {
            **(preset.extra_config or {}),
            "rollout_max_context_len": context,
        }
    return fields


def recipe_for(model: ModelConfig, max_context_len: int | None = None) -> Recipe:
    """Return the preset recipe that trains ``model`` at ``max_context_len``.

    Picks the shortest-context preset that fits, then sets its rollout limits
    to ``max_context_len``. ``None`` keeps the preset's own limits.
    """
    ctx = max_context_len

    def fits(limit: int) -> bool:
        return ctx is None or ctx <= limit

    # Cases match in order: a LoRA subclass must come before its base model.
    match model:
        # Presets built for a fixed context: the shortest one that fits.
        case models.Kimi_K3() if fits(32_768):
            preset = miles_recipe.Kimi_K3_LoRA_Recipe()
        case models.Kimi_K3() if fits(65_536):
            preset = miles_recipe.Kimi_K3_LoRA_64k_Recipe()
        case models.GLM_5_3_Flash_LoRA() if fits(32_768):
            preset = miles_recipe.GLM_5_3_Flash_LoRA_Recipe()
        case models.GLM_5_3_Flash_LoRA() if fits(262_144):
            preset = miles_recipe.GLM_5_3_Flash_LoRA_256k_Recipe()
        case models.Beam_1_0_501B_A23B_LoRA() if fits(262_144):
            preset = miles_recipe.Beam_1_0_501B_A23B_LoRA_256k_Recipe()
        case models.Muse_Spark_Open() if fits(65_536):
            preset = miles_recipe.Muse_Spark_Open_64k_Recipe()
        case (
            models.Kimi_K3()
            | models.GLM_5_3_Flash_LoRA()
            | models.Beam_1_0_501B_A23B_LoRA()
            | models.Muse_Spark_Open()
        ):
            raise DojoConfigError(
                f"No {type(model).__name__} preset supports max_context_len={ctx}"
            )
        # Presets that adapt to any context. slime:
        case models.GLM_4_7():
            preset = slime_recipe.GLM_4_7_Recipe()
        case models.Qwen3_0_6B():
            preset = slime_recipe.Qwen3_0_6B_Recipe()
        case models.Qwen3_1_7B():
            preset = slime_recipe.Qwen3_1_7B_Recipe()
        case models.Qwen3_4B():
            preset = slime_recipe.Qwen3_4B_Recipe()
        case models.Qwen3_8B():
            preset = slime_recipe.Qwen3_8B_Recipe()
        case models.Qwen3_5_0_8B():
            preset = slime_recipe.Qwen3_5_0_8B_Recipe()
        case models.Qwen3_5_2B():
            preset = slime_recipe.Qwen3_5_2B_Recipe()
        case models.Qwen3_5_4B():
            preset = slime_recipe.Qwen3_5_4B_Recipe()
        case models.Qwen3_5_9B():
            preset = slime_recipe.Qwen3_5_9B_Recipe()
        case models.Qwen3_6_27B():
            preset = slime_recipe.Qwen3_6_27B_Recipe()
        case models.Qwen3_6_35B():
            preset = slime_recipe.Qwen3_6_35B_Recipe()
        case models.Qwen3_8_27B():
            preset = slime_recipe.Qwen3_8_27B_Recipe()
        case models.Qwen3_ASR_1_7B():
            preset = slime_recipe.Qwen3_ASR_1_7B_Recipe()
        case models.Qwen3_VL_8B():
            preset = slime_recipe.Qwen3_VL_8B_Recipe()
        # Miles:
        case models.DeepSeek_V4_1_Flash_LoRA():
            preset = miles_recipe.DeepSeek_V4_1_Flash_LoRA_Recipe()
        case models.DeepSeek_V4_1_Flash():
            preset = miles_recipe.DeepSeek_V4_1_Flash_Recipe()
        case models.Gemma4_26B_A4B():
            preset = miles_recipe.Gemma4_26B_A4B_Recipe()
        case models.Inkling_Small_LoRA():
            preset = miles_recipe.Inkling_Small_LoRA_Recipe()
        case models.Inkling_Small():
            preset = miles_recipe.Inkling_Small_Recipe()
        case models.Moonlight_16B_A3B_Instruct():
            preset = miles_recipe.Moonlight_16B_A3B_Recipe()
        case _:
            raise DojoConfigError(
                f"No recipe preset for {type(model).__name__} "
                f"({model.model_name!r}). Use TrainConfig with an explicit "
                "recipe for this model."
            )
    if ctx is None:
        return preset
    return dataclasses.replace(preset, **_context_fields(preset, ctx))


# TODO: add to the API reference once the API shape is final.
@exclude_from_api_reference
class RoutedTrainConfig:
    """Route a model, environment, and budget to the best recipe we have.

    The user states what to train (model, environment with its context length)
    and when to stop (budget); picking a framework and preset is ours, so recipe
    improvements reach every routed run without a script change.

    ```python
    run = RoutedTrainConfig(
        Qwen3_5_4B(),
        Environment(
            dataset=my_dataset, reward_fn=my_reward, max_context_len=32_768
        ),
        Budget(max_cost_usd=50, max_hours=6),
    ).train()
    ```

    The result is an ordinary ``TrainConfig``; edit it before training to
    change any hyperparameter:

    ```python
    routed = RoutedTrainConfig(Qwen3_5_4B(), env, Budget(max_rollouts=100))
    routed.train_config.recipe = dataclasses.replace(routed.recipe, lr=5e-7)
    run = routed.train()
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

    def _build_recipe(self) -> Recipe:
        env = self.environment
        preset = recipe_for(self.model, env.max_context_len)
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
        num_tasks = len(list(env.dataset.rows()))
        if num_tasks < preset.rollout_batch_size:
            samples = overrides.get("n_samples_per_prompt", preset.n_samples_per_prompt)
            derived["rollout_batch_size"] = num_tasks
            derived["global_batch_size"] = num_tasks * samples
        return dataclasses.replace(preset, **{**derived, **overrides})

    @property
    def recipe(self) -> Recipe:
        return self.train_config.recipe

    def train(self, *, show_output: bool = True) -> TrainingRun:
        return self.train_config.train(show_output=show_output)

    def launch(self, *, show_output: bool = True) -> TrainingRun:
        return self.train_config.launch(show_output=show_output)
