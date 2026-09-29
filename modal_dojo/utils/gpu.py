"""Shared GPU type aliases."""

from typing import Literal

# Supported Modal GPU families across all frameworks. Framework configs
# that accept a GPU name should annotate it with this type.
GPUType = Literal["H100", "H200", "B200", "B300"]

__all__ = ["GPUType"]
