"""Helpers for tinker-cookbook configs that SpindleRecipe runs."""

from __future__ import annotations

from importlib import import_module
from typing import Any

# These need modal-dojo[spindle], so they load on first use.
_LAZY = {
    "cookbook_rl": "modal_dojo.frameworks.spindle.cookbook",
    "rl_dataset_builder": "modal_dojo.frameworks.spindle.dataset_builders",
}

__all__ = [
    "cookbook_rl",
    "rl_dataset_builder",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
