"""slime framework package.

``build_slime_app`` is re-exported lazily so importing a sibling module (e.g. the
in-image transcription rollout) doesn't pull in the launcher, which imports
``modal`` — unavailable in the slime training image.
"""

from __future__ import annotations

__all__ = ["build_slime_app"]


def __getattr__(name: str):
    if name == "build_slime_app":
        from .launcher import build_slime_app

        return build_slime_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
