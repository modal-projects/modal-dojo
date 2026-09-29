"""Reusable Click options and confirmation helpers."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TypeVar

import click


CommandFunction = TypeVar("CommandFunction", bound=Callable[..., Any])


def json_option(function: CommandFunction) -> CommandFunction:
    return click.option(
        "-j",
        "--json",
        "json_output",
        is_flag=True,
        default=False,
        help="Output machine-readable JSON.",
    )(function)


def yes_option(function: CommandFunction) -> CommandFunction:
    return click.option(
        "-y",
        "--yes",
        "--force",
        is_flag=True,
        default=False,
        help="Run without pausing for confirmation.",
    )(function)


def confirm_or_require_yes(message: str) -> None:
    """Prompt interactively, or direct non-interactive callers to ``--yes``."""
    if not sys.stdin.isatty():
        raise click.UsageError(
            "No interactive terminal detected; rerun with --yes (-y)."
        )
    click.confirm(message, default=False, abort=True)
