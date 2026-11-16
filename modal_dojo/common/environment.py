"""An RL environment: a dataset and how responses to it are scored."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.dataset import DatasetConfig


# TODO: add to the API reference once the API shape is final.
@exclude_from_api_reference
class Environment:
    """A training dataset and the reward function that scores responses to it.

    Attributes:
        dataset: Prompts and labels to train on.
        reward_fn: The recipe's ``custom_rm_function``.
        generate_fn: Optional ``custom_generate_function`` for multi-turn rollouts.
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
        return self._dataset

    @property
    def reward_fn(self) -> Callable[..., Any]:
        return self._reward_fn

    @property
    def generate_fn(self) -> Callable[..., Any] | None:
        return self._generate_fn
