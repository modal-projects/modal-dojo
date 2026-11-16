"""An RL environment: the tasks a model trains on and how its answers are scored."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modal_dojo.common.dataset import DatasetConfig


class Environment:
    """A training dataset and the reward function that scores responses to it.

    ```python
    def reward_fn(args, sample, **kwargs) -> float:
        return float(sample.response.strip() == sample.label)

    env = Environment(dataset=my_dataset, reward_fn=reward_fn)
    run = ManagedTrainConfig(Qwen3_5_4B(), env, num_rollout=20).train()
    ```

    Subclasses package a source of tasks with its own scoring, and may build
    their dataset and functions lazily by overriding the properties.

    Attributes:
        dataset: Prompts and labels to train on.
        reward_fn: Reward for one response, with the recipe
            ``custom_rm_function`` signature ``(args, sample, **kwargs)``.
        generate_fn: Optional rollout, with the recipe
            ``custom_generate_function`` signature ``(args, sample,
            sampling_params)``, for environments the model interacts with over
            several turns. ``None`` keeps the framework's own generation.
    """

    def __init__(
        self,
        dataset: DatasetConfig,
        reward_fn: Callable[..., Any],
        generate_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._dataset = dataset
        self._reward_fn = reward_fn
        self._generate_fn = generate_fn

    @property
    def dataset(self) -> DatasetConfig:
        """Prompts and labels to train on."""
        return self._dataset

    @property
    def reward_fn(self) -> Callable[..., Any]:
        """The recipe's ``custom_rm_function``."""
        return self._reward_fn

    @property
    def generate_fn(self) -> Callable[..., Any] | None:
        """The recipe's ``custom_generate_function``, or ``None``."""
        return self._generate_fn
