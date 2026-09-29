from __future__ import annotations

import inspect
from enum import Enum
from typing import Any

import modal_dojo as gym
from modal_dojo._api_reference import is_excluded_from_api_reference
from scripts.generate_models_table import collect_model_preset_names

API_REFERENCE_DENYLIST: frozenset[str] = frozenset(
    name
    for name in gym.__all__
    if is_excluded_from_api_reference(name, getattr(gym, name))
)

GROUPS = {
    "models": {"label": "Models", "order": 1},
    "datasets": {"label": "Datasets", "order": 2},
    "recipes": {"label": "Recipes", "order": 3},
    "training": {"label": "Training", "order": 4},
    "deployment": {"label": "Deployment", "order": 5},
}


def _group_for_module(module: str) -> str:
    if ".models" in module:
        return "models"
    if ".dataset" in module:
        return "datasets"
    if "train_recipes" in module:
        return "recipes"
    if (
        "deploy_recipes" in module
        or module.endswith(".deployment")
        or module.endswith(".endpoint")
        or module.endswith(".sandbox")
    ):
        return "deployment"
    return "training"


def _kind(obj: Any) -> str | None:
    if inspect.isclass(obj):
        return "enum" if issubclass(obj, Enum) else "class"
    if inspect.isfunction(obj) or inspect.isroutine(obj):
        return "function"
    return None


def entry_sort_key(entry: dict[str, str]) -> tuple[bool, str]:
    return (entry["kind"] == "function", entry["class_name"].casefold())


def collect_public_api() -> list[dict[str, Any]]:
    presets = collect_model_preset_names()
    entries: list[dict[str, str]] = []
    for name in gym.__all__:
        if name in API_REFERENCE_DENYLIST:
            continue
        obj = getattr(gym, name)
        kind = _kind(obj)
        if kind is None:
            continue
        entries.append(
            {
                "class_name": name,
                "module": gym.__name__,
                "group": _group_for_module(obj.__module__),
                "sidebar_label": name,
                "kind": kind,
                "sidebar_excluded": name in presets,
            }
        )
    entries.sort(
        key=lambda entry: (
            GROUPS[entry["group"]]["order"],
            *entry_sort_key(entry),
        )
    )
    return entries


API_REFERENCE_MANIFEST = collect_public_api()

CLASS_REFERENCE_PATHS = {
    entry["class_name"]: f"/reference/{entry['class_name'].lower()}/"
    for entry in API_REFERENCE_MANIFEST
}
