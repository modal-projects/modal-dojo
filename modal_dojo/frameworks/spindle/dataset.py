"""Explicit adapters from Dojo data sources to cookbook RL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from tinker_cookbook.rl.types import RLDatasetBuilder


@exclude_from_api_reference
class DojoDatasetBuilder:
    """Convert an opted-in Dojo dataset into a Tinker Cookbook RL builder."""

    @staticmethod
    def from_dataset(dataset: DatasetConfig, **kwargs: Any) -> RLDatasetBuilder:
        """Convert immediately, so unsupported sources fail before deployment.

        ``from`` is a Python keyword; use ``from_dataset``. Custom sources opt in
        by implementing ``DatasetConfig.to_dataset_builder``. Conversion does
        not download data or allocate GPUs.
        """
        if not isinstance(dataset, DatasetConfig):
            raise DojoConfigError("Expected a DatasetConfig")
        return dataset.to_dataset_builder(**kwargs)


def huggingface_builder(
    dataset, *, model_name, renderer_name, batch_size, group_size, reward_fn
):
    if dataset.input_format == "raw":
        raise DojoConfigError(
            "Cookbook conversion supports text/messages, not raw inputs"
        )
    if dataset.label_key() is None:
        raise DojoConfigError(
            "Cookbook conversion requires an output_column for rewards"
        )
    if not callable(reward_fn):
        raise DojoConfigError("reward_fn must score (response_text, label) -> float")
    _validate_sizes(batch_size, group_size)
    from .dataset_builders import HuggingFaceBuilder

    return HuggingFaceBuilder(
        dataset=dataset,
        model_name=model_name,
        renderer_name=renderer_name,
        batch_size=batch_size,
        group_size=group_size,
        reward_fn=reward_fn,
    )


def harbor_builder(
    dataset, *, model_name, renderer_name, batch_size, group_size, **kwargs
):
    _validate_sizes(batch_size, group_size)
    if dataset.system_prompt:
        raise DojoConfigError(
            "Cookbook Harbor provides its own system prompt; omit system_prompt"
        )
    # Validate cookbook options at conversion, before loading tasks or deploying.
    import chz
    from tinker_cookbook.recipes.harbor_rl.harbor_env import HarborDatasetBuilder
    from .dataset_builders import HarborBuilder

    unknown = set(kwargs) - (
        set(chz.chz_fields(HarborDatasetBuilder))
        - {
            "tasks",
            "batch_size",
            "group_size",
            "model_name",
            "renderer_name",
        }
    )
    if unknown:
        raise DojoConfigError(f"Unknown cookbook Harbor options: {sorted(unknown)}")
    return HarborBuilder(
        dataset=dataset,
        model_name=model_name,
        renderer_name=renderer_name,
        batch_size=batch_size,
        group_size=group_size,
        options=kwargs,
    )


def _validate_sizes(batch_size, group_size):
    if batch_size < 1 or group_size < 1:
        raise DojoConfigError("batch_size and group_size must be positive")
