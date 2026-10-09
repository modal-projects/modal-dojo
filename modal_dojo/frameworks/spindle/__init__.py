"""Spindle framework package. ``build_spindle_app`` is re-exported lazily so the
in-container GRPO driver can be imported without ``modal``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .launcher import build_spindle_app

__all__ = ["build_spindle_app"]


def __getattr__(name: str):
    if name == "build_spindle_app":
        from .launcher import build_spindle_app

        return build_spindle_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
