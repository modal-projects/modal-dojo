#!/usr/bin/env python3
"""Report which model/config classes and tutorials are affected by a diff.

Usage:
    git diff --cached | uv run python -m scripts.diff_impact
    uv run python -m scripts.diff_impact --diff-file /tmp/change.diff
    uv run python -m scripts.diff_impact < /tmp/change.diff

The script reads a unified diff, extracts changed paths, and maps those paths
to public classes and directly changed tutorials.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from scripts.tutorial_index import TutorialEntry, load_tutorial_index

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_SRC_ROOT = REPO_ROOT / "tutorials"
MODEL_PACKAGE_ROOT = REPO_ROOT / "modal_dojo" / "common" / "models"
SLIME_RECIPE_PACKAGE_ROOT = REPO_ROOT / "modal_dojo" / "train_recipes" / "slime_recipe"
MILES_RECIPE_PACKAGE_ROOT = REPO_ROOT / "modal_dojo" / "train_recipes" / "miles_recipe"

VALIDATION_BACKEND_ROOT = REPO_ROOT / "scripts" / "validation_backends"

# Touching the shared validation harness or shared training plumbing
# invalidates every model, so a diff that hits any of these forces a full
# re-validation.
SHARED_VALIDATION_HARNESS_PATHS = frozenset(
    {
        REPO_ROOT / "scripts" / "validate_model_configs.py",
        REPO_ROOT / "scripts" / "diff_impact.py",
        VALIDATION_BACKEND_ROOT / "__init__.py",
        REPO_ROOT / "modal_dojo" / "common" / "models" / "validation.py",
        REPO_ROOT / "modal_dojo" / "common" / "train.py",
        REPO_ROOT / "modal_dojo" / "common" / "train_result.py",
    }
)

# Per-framework harness paths invalidate only the models that train on that
# framework. Without this split, adding a miles model would make every
# miles-only change re-validate the whole slime set.
FRAMEWORK_VALIDATION_HARNESS_PATHS: dict[str, frozenset[Path]] = {
    "slime": frozenset(
        {
            REPO_ROOT / "modal_dojo" / "frameworks" / "slime" / "launcher.py",
            VALIDATION_BACKEND_ROOT / "slime.py",
        }
    ),
    "miles": frozenset(
        {
            REPO_ROOT / "modal_dojo" / "frameworks" / "miles" / "launcher.py",
            VALIDATION_BACKEND_ROOT / "miles.py",
        }
    ),
}


@dataclass(frozen=True)
class ImpactReport:
    affected_classes: tuple[str, ...]
    affected_tutorials: tuple[tuple[str, str, tuple[str, ...]], ...]


@lru_cache(maxsize=1)
def _load_tutorial_index() -> dict[str, TutorialEntry]:
    return {tutorial.slug: tutorial for tutorial in load_tutorial_index()}


def _parse_public_definitions(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
    return names


@lru_cache(maxsize=None)
def _package_public_definitions(package_root: str) -> set[str]:
    root = Path(package_root)
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        names.update(_parse_public_definitions(path))
    return names


def _base_recipe_for(framework, model_config):
    """The framework's base recipe for a model, importing only that framework.

    Same rule as ``validation_backends.build_recipe_and_dataset``: each
    framework is imported inside its own branch, so indexing the slime models
    that gate PRs never imports the miles recipes — a broken miles recipe must
    not be able to fail every pull request.
    """
    from modal_dojo.common.models.validation import Framework

    if framework is Framework.SLIME:
        from modal_dojo.train_recipes.slime_recipe import SlimeRecipe

        return SlimeRecipe.get_base_recipe(model_config)
    if framework is Framework.MILES:
        from modal_dojo.train_recipes.miles_recipe import MilesRecipe

        return MilesRecipe.get_base_recipe(model_config)
    raise ValueError(f"no base recipe lookup for framework {framework!r}")


@lru_cache(maxsize=1)
def _model_index() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Map defining classes to the model names they gate, and frameworks to theirs.

    A model is validated by ``validate_model_configs.py check --model <name>``,
    where ``<name>`` is the registry's short name (e.g. ``Qwen3-0.6B``). Both
    the model's ``ModelConfig`` subclass and the recipe class its framework
    returns from ``get_base_recipe`` gate that model, so a change to either
    re-validates it.
    """
    from modal_dojo.common.models.validation import _ValidationConfig

    class_to_models: dict[str, set[str]] = defaultdict(set)
    framework_to_models: dict[str, set[str]] = defaultdict(set)
    for config in _ValidationConfig.select():
        class_to_models[config.model_config.__name__].add(config.name)
        framework_to_models[config.framework.value].add(config.name)
        recipe = _base_recipe_for(config.framework, config.model_config())
        if recipe is not None:
            class_to_models[type(recipe).__name__].add(config.name)

    return (
        {name: frozenset(models) for name, models in class_to_models.items()},
        {name: frozenset(models) for name, models in framework_to_models.items()},
    )


def affected_models(diff_text: str) -> tuple[str, ...]:
    """Model names (validate ``--model`` args) impacted by a diff.

    Importing ``modal_dojo`` is deferred to here so the tutorial-only
    paths through ``analyze_diff`` stay import-free.
    """
    changed_paths = _paths_from_diff(diff_text)
    class_to_models, framework_to_models = _model_index()
    all_models = {model for models in framework_to_models.values() for model in models}

    touched_frameworks = {
        framework
        for framework, harness_paths in FRAMEWORK_VALIDATION_HARNESS_PATHS.items()
        if any(path in harness_paths for path in changed_paths)
    }
    if any(path in SHARED_VALIDATION_HARNESS_PATHS for path in changed_paths):
        return tuple(sorted(all_models))

    models: set[str] = set()
    for framework in touched_frameworks:
        models.update(framework_to_models.get(framework, frozenset()))

    report = analyze_diff(diff_text)
    for class_name in report.affected_classes:
        models.update(class_to_models.get(class_name, frozenset()))
    return tuple(sorted(models))


def _paths_from_diff(diff_text: str) -> set[Path]:
    paths: set[Path] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if not match:
                continue
            for raw in match.groups():
                if raw != "/dev/null":
                    paths.add(REPO_ROOT / raw)
            continue

        if line.startswith("rename from ") or line.startswith("rename to "):
            raw = line.split(" ", 2)[2].strip()
            if raw != "/dev/null":
                paths.add(REPO_ROOT / raw)
            continue

        if line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            if raw != "/dev/null":
                paths.add(REPO_ROOT / raw)

    return paths


def _classes_for_path(path: Path) -> set[str]:
    if path.is_relative_to(MODEL_PACKAGE_ROOT):
        if path.name in {"base.py", "__init__.py"}:
            return set(_package_public_definitions(str(MODEL_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.is_relative_to(SLIME_RECIPE_PACKAGE_ROOT):
        if path.name in {"recipe.py", "__init__.py"}:
            return set(_package_public_definitions(str(SLIME_RECIPE_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.is_relative_to(MILES_RECIPE_PACKAGE_ROOT):
        if path.name in {"recipe.py", "__init__.py"}:
            return set(_package_public_definitions(str(MILES_RECIPE_PACKAGE_ROOT)))
        return _parse_public_definitions(path)

    if path.suffix == ".py" and path.exists():
        return _parse_public_definitions(path)

    return set()


def _tutorial_slug_for_path(
    path: Path,
    tutorials: dict[str, TutorialEntry],
) -> str | None:
    if not path.is_relative_to(TUTORIAL_SRC_ROOT):
        return None
    for slug, tutorial in tutorials.items():
        owner = (
            tutorial.path
            if tutorial.path.parent == TUTORIAL_SRC_ROOT
            else tutorial.path.parent
        )
        if path == owner or path.is_relative_to(owner):
            return slug
    return None


def analyze_diff(diff_text: str) -> ImpactReport:
    tutorials = _load_tutorial_index()
    changed_paths = _paths_from_diff(diff_text)

    affected_classes: set[str] = set()
    affected_tutorial_reasons: dict[str, set[str]] = defaultdict(set)

    for path in changed_paths:
        if path.is_relative_to(TUTORIAL_SRC_ROOT):
            slug = _tutorial_slug_for_path(path, tutorials)
            if slug is not None:
                affected_tutorial_reasons[slug].add("tutorial source changed")
            continue

        classes = _classes_for_path(path)
        affected_classes.update(classes)

    affected_tutorials = tuple(
        sorted(
            (
                slug,
                tutorials[slug].title,
                tuple(sorted(reasons)),
            )
            for slug, reasons in affected_tutorial_reasons.items()
            if slug in tutorials
        )
    )

    return ImpactReport(
        affected_classes=tuple(sorted(affected_classes)),
        affected_tutorials=affected_tutorials,
    )


def _format_report(report: ImpactReport) -> str:
    lines: list[str] = []
    lines.append("Affected classes:")
    if report.affected_classes:
        for name in report.affected_classes:
            lines.append(f"- {name}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Tutorials to rerun:")
    if report.affected_tutorials:
        for slug, summary, reasons in report.affected_tutorials:
            reason_text = ", ".join(reasons)
            if reason_text:
                lines.append(f"- {slug} — {summary} ({reason_text})")
            else:
                lines.append(f"- {slug} — {summary}")
    else:
        lines.append("- none")

    return "\n".join(lines)


def _report_to_json(report: ImpactReport) -> str:
    payload = {
        "affected_classes": list(report.affected_classes),
        "affected_tutorials": [
            {
                "slug": slug,
                "summary": summary,
                "reasons": list(reasons),
            }
            for slug, summary, reasons in report.affected_tutorials
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a diff and report affected model classes and tutorials."
    )
    parser.add_argument(
        "--diff-file",
        type=Path,
        help="Read the diff from a file instead of stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable summary.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Emit a JSON array of model names affected by the diff, "
        "compatible with scripts/validate_model_configs.py check --model.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.diff_file is not None:
        diff_text = args.diff_file.read_text()
    else:
        diff_text = sys.stdin.read()

    if args.models:
        print(json.dumps(list(affected_models(diff_text))))
        return 0

    report = analyze_diff(diff_text)
    output = _report_to_json(report) if args.json else _format_report(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
