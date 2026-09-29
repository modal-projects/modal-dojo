"""Small, TTY-aware output helpers for CLI commands."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Column, Table
from rich.text import Text


FALLBACK_OUTPUT_WIDTH = 120


def _console(*, stderr: bool = False) -> Console:
    width = shutil.get_terminal_size(fallback=(FALLBACK_OUTPUT_WIDTH, 24)).columns
    rich_enabled = sys.stdout.isatty()
    return Console(
        stderr=stderr,
        highlight=False,
        force_jupyter=False,
        force_terminal=rich_enabled,
        color_system="standard" if rich_enabled else None,
        width=width,
    )


def print_json(value: Any, *, stderr: bool = False) -> None:
    """Write JSON without Rich formatting."""
    print(
        json.dumps(value, ensure_ascii=False, indent=2),
        file=sys.stderr if stderr else sys.stdout,
    )


def print_table(
    columns: Sequence[Column | str],
    rows: Sequence[Sequence[object]],
    *,
    title: str = "",
    show_header: bool = True,
) -> None:
    safe_columns = [
        Column(header=Text(column)) if isinstance(column, str) else column
        for column in columns
    ]
    table = Table(
        *safe_columns,
        title=Text(title) if title else None,
        show_header=show_header,
    )
    for row in rows:
        cells = [
            cell if cell is None or isinstance(cell, Text) else Text(str(cell))
            for cell in row
        ]
        table.add_row(*cells)
    _console().print(table)


def print_renderable(renderable: RenderableType) -> None:
    """Render a Rich object to stdout using the shared terminal settings."""
    _console().print(renderable)


def print_error(message: str) -> None:
    """Write an error to stderr when Rich output is enabled."""
    console = _console(stderr=True)
    if sys.stdout.isatty():
        console.print(
            Panel(
                Text(message),
                title="Error",
                title_align="left",
                border_style="red",
                expand=True,
            )
        )
    else:
        console.print(message, markup=False)


def print_warning(message: str) -> None:
    Console(stderr=True, highlight=False).print(
        f"[bold yellow]Warning:[/bold yellow] {escape(message)}"
    )
