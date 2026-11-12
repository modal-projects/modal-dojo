"""Helpers for tinker-cookbook configs that SpindleRecipe runs."""

from __future__ import annotations

from typing import Any

# Needs modal-dojo[spindle].
_BUILDERS = ("rl_dataset_builder",)

__all__ = [
    "rl_dataset_builder",
]


def __getattr__(name: str) -> Any:
    if name in _BUILDERS:
        from modal_dojo.frameworks.spindle import dataset_builders

        return getattr(dataset_builders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
