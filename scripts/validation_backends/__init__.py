"""Per-framework halves of the model validation harness.

``scripts/validate_model_configs.py`` owns everything that is the same for
every framework: the CLI, the result dataclass, the markdown summary and the
PR comment. The framework module supplies which recipe
trains the model and which dataset it trains on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.validation import Framework

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfig
    from modal_training_gym.train_recipes.base import BaseTrainRecipe


def build_recipe_and_dataset(
    framework: Framework,
    model_config: "ModelConfig",
    step_count: int,
    *,
    loss_type: str = "policy_loss",
) -> tuple["BaseTrainRecipe", "DatasetConfig"]:
    """The model's base recipe and the dataset it validates against.

    Each framework is imported inside its own branch rather than at module
    scope so validating a slime model never imports the miles recipes: a broken
    miles backend must not be able to take down the slime validation that gates
    PRs.
    """
    if framework is Framework.SLIME:
        from .slime import build_slime_validation

        return build_slime_validation(model_config, step_count, loss_type=loss_type)
    if framework is Framework.MILES:
        from .miles import build_miles_validation

        return build_miles_validation(model_config, step_count, loss_type=loss_type)
    raise TrainingGymConfigError(f"no validation backend for framework {framework!r}")
