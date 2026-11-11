"""Tinker code that SpindleRecipe runs on Modal against Spindle."""

from __future__ import annotations

from typing import Any

from modal_dojo.tinker.program import TinkerContext, TinkerProgram

# tinker-cookbook dataset builders; they need modal-dojo[spindle].
_BUILDERS = ("harbor_dataset_builder", "huggingface_dataset_builder")

__all__ = [
    "TinkerContext",
    "TinkerProgram",
    "harbor_dataset_builder",
    "huggingface_dataset_builder",
]


def __getattr__(name: str) -> Any:
    if name in _BUILDERS:
        from modal_dojo.frameworks.spindle import dataset_builders

        return getattr(dataset_builders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
