from __future__ import annotations

from typing import Any, TypeVar, overload

T = TypeVar("T")

_EXCLUDE_ATTR = "_exclude_from_api_reference"
_EXCLUDED_EXPORTS: set[str] = set()


@overload
def exclude_from_api_reference(obj: T) -> T: ...


@overload
def exclude_from_api_reference(obj: str) -> None: ...


def exclude_from_api_reference(obj: T | str) -> T | None:
    """Omit ``obj`` from generated SDK reference pages.

    Call with a class or function to mark that object. Call with an export
    name when the public name is an alias of a documented type.
    """
    if isinstance(obj, str):
        _EXCLUDED_EXPORTS.add(obj)
        return None
    setattr(obj, _EXCLUDE_ATTR, True)
    return obj


def is_excluded_from_api_reference(name: str, obj: Any) -> bool:
    return name in _EXCLUDED_EXPORTS or bool(getattr(obj, _EXCLUDE_ATTR, False))
