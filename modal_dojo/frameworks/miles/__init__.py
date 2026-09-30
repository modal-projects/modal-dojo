"""miles framework package.

``build_miles_app`` is re-exported lazily so importing a sibling module (e.g.
the in-container ``phase_reporting`` hooks) doesn't pull in the launcher, which
imports ``modal`` — unavailable in the miles training image.
"""

from __future__ import annotations

__all__ = ["build_miles_app"]


def __getattr__(name: str):
    if name == "build_miles_app":
        from .launcher import build_miles_app

        return build_miles_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
