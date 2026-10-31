"""Explicit adapters from Dojo data sources to tinker-cookbook RL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modal_dojo._api_reference import exclude_from_api_reference
from modal_dojo.common.dataset import DatasetConfig
from modal_dojo.common.errors import DojoConfigError

if TYPE_CHECKING:
    from tinker_cookbook.rl.types import RLDatasetBuilder


@exclude_from_api_reference
class DojoDatasetBuilder:
    """Convert an opted-in Dojo dataset into a tinker-cookbook RL builder."""

    @staticmethod
    def from_dataset(dataset: DatasetConfig, **kwargs: Any) -> RLDatasetBuilder:
        """Convert via DatasetConfig.to_dataset_builder, rejecting unsupported sources."""
        if not isinstance(dataset, DatasetConfig):
            raise DojoConfigError("Expected a DatasetConfig")
        return dataset.to_dataset_builder(**kwargs)
