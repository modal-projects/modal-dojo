"""Modal-style Click help and command classes for training-gym. Ported over almost entirely from Modal's helpers."""

from __future__ import annotations

import inspect
import shutil
import sys
from typing import Any

import click
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


_HEADING_STYLE = "bold bright_green"
_COMMAND_NAME_STYLE = ""
_COMMAND_DESC_STYLE = "dim"
_OPTION_FLAG_STYLE = "green"
_OPTION_METAVAR_STYLE = "dim"
_ERROR_STYLE = "red"

_MAX_HELP_WIDTH = 80
_HELP_PADDING = 1


def use_rich_style() -> bool:
    """Whether help and errors should be rendered in Rich style."""
    return sys.stdout.isatty()


def _make_help_console() -> Console:
    columns, _ = shutil.get_terminal_size()
    return Console(highlight=False, width=min(_MAX_HELP_WIDTH, columns))


def _build_usage(cmd: click.Command, ctx: click.Context) -> RenderableType:
    pieces = " ".join(cmd.collect_usage_pieces(ctx))
    usage = Text()
    usage.append("Usage: ", style=_HEADING_STYLE)
    usage.append(f"{ctx.command_path} {pieces}".rstrip())
    return usage


def _build_help_text(cmd: click.Command) -> RenderableType | None:
    text = cmd.help or cmd.short_help or ""
    if not text:
        return None
    return Markdown(inspect.cleandoc(text))


def _option_label(param: click.Parameter, ctx: click.Context) -> Text:
    text = Text()
    for index, option in enumerate(param.opts):
        if index > 0:
            text.append(", ")
        text.append(option, style=_OPTION_FLAG_STYLE)
    if param.secondary_opts:
        text.append(" / ")
        for index, option in enumerate(param.secondary_opts):
            if index > 0:
                text.append(", ")
            text.append(option, style=_OPTION_FLAG_STYLE)
    if not getattr(param, "is_flag", False) and not getattr(param, "count", False):
        text.append(" ")
        if "ctx" in inspect.signature(click.Parameter.make_metavar).parameters:
            metavar = param.make_metavar(ctx)
        else:
            metavar = param.make_metavar()  # type: ignore[call-arg]
        text.append(metavar, style=_OPTION_METAVAR_STYLE)
    return text


def _build_options(cmd: click.Command, ctx: click.Context) -> RenderableType | None:
    rows: list[tuple[Text, str]] = []
    for param in cmd.get_params(ctx):
        record = param.get_help_record(ctx)
        if record is None:
            continue
        rows.append((_option_label(param, ctx), record[1] or ""))
    if not rows:
        return None

    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for label, help_text in rows:
        table.add_row(label, help_text)
    return Group(Text("Options", style=_HEADING_STYLE), table)


def _build_epilog(cmd: click.Command) -> RenderableType | None:
    if not cmd.epilog:
        return None
    return Text(cmd.epilog)


def group_commands_by_panel(
    group: click.Group,
) -> dict[str, list[tuple[str, click.Command]]]:
    """Bucket visible subcommands, preserving registration order."""
    panels: dict[str, list[tuple[str, click.Command]]] = {}
    for name, command in group.commands.items():
        if command.hidden:
            continue
        panel = getattr(command, "panel", None) or "Commands"
        panels.setdefault(panel, []).append((name, command))
    return panels


def _build_commands(group: click.Group, available_width: int) -> RenderableType | None:
    panels = group_commands_by_panel(group)
    if not panels:
        return None

    name_width = max(len(name) for items in panels.values() for name, _ in items)

    parts: list[RenderableType] = []
    for panel_name, items in panels.items():
        if parts:
            parts.append(Text(""))
        parts.append(
            Text(
                panel_name.ljust(available_width),
                style=f"{_HEADING_STYLE} underline",
            )
        )
        parts.append(_build_command_table(items, name_width, available_width))
    return Group(*parts)


def build_command_table(name_width: int, table_width: int | None = None) -> Table:
    """Build a Modal-style name and description table."""
    kwargs: dict[str, Any] = {
        "box": None,
        "show_header": False,
        "pad_edge": True,
        "padding": (0, 1),
    }
    if table_width is not None:
        kwargs["width"] = table_width
    table = Table(**kwargs)
    table.add_column(
        style=_COMMAND_NAME_STYLE,
        no_wrap=True,
        width=name_width,
    )
    table.add_column(
        style=_COMMAND_DESC_STYLE,
        overflow="fold",
        ratio=1,
    )
    return table


def _build_command_table(
    items: list[tuple[str, click.Command]],
    name_width: int,
    table_width: int,
) -> Table:
    table = build_command_table(name_width, table_width)
    for name, command in items:
        table.add_row(name, command.get_short_help_str(limit=80))
    return table


def _emit(
    console: Console,
    sections: list[RenderableType | None],
    formatter: click.HelpFormatter,
) -> None:
    parts: list[RenderableType] = []
    for section in sections:
        if section is None:
            continue
        if parts:
            parts.append(Text(""))
        parts.append(section)
    if parts:
        parts.append(Text(""))
    with console.capture() as capture:
        console.print(Padding(Group(*parts), (0, _HELP_PADDING)))
    formatter.write(capture.get())


class _TrainingGymCommand(click.Command):
    """Click command that renders ``--help`` with custom Rich output."""

    def __init__(
        self,
        *args: Any,
        panel: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.panel = panel

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not use_rich_style():
            return super().format_help(ctx, formatter)
        console = _make_help_console()
        _emit(
            console,
            [
                _build_usage(self, ctx),
                _build_help_text(self),
                _build_options(self, ctx),
                _build_epilog(self),
            ],
            formatter,
        )


class _TrainingGymGroup(click.Group):
    """Click group whose commands and nested groups share custom help."""

    command_class = _TrainingGymCommand
    group_class = type

    def __init__(
        self,
        *args: Any,
        panel: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("no_args_is_help", True)
        super().__init__(*args, **kwargs)
        self.panel = panel

    def add_command(
        self,
        cmd: click.Command,
        name: str | None = None,
        *,
        panel: str | None = None,
        hidden: bool | None = None,
    ) -> None:
        super().add_command(cmd, name)
        if panel is not None:
            cmd.panel = panel  # type: ignore[attr-defined]
        if hidden is not None:
            cmd.hidden = hidden

    def format_commands(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        for panel_name, items in group_commands_by_panel(self).items():
            rows = [
                (name, command.get_short_help_str(limit=80)) for name, command in items
            ]
            with formatter.section(panel_name):
                formatter.write_dl(rows)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not use_rich_style():
            return super().format_help(ctx, formatter)
        console = _make_help_console()
        _emit(
            console,
            [
                _build_usage(self, ctx),
                _build_help_text(self),
                _build_options(self, ctx),
                _build_commands(
                    self,
                    min(_MAX_HELP_WIDTH, console.width) - _HELP_PADDING * 2,
                ),
                _build_epilog(self),
            ],
            formatter,
        )


def _render_click_exception(exc: click.ClickException, file: Any) -> None:
    console = Console(
        file=file if file is not None else sys.stderr,
        highlight=False,
    )

    if isinstance(exc, click.UsageError) and exc.ctx is not None:
        ctx = exc.ctx
        console.print(ctx.get_usage())
        if ctx.command.get_help_option(ctx) is not None:
            option = ctx.help_option_names[0] if ctx.help_option_names else "--help"
            console.print(f"Try [bold]'{ctx.command_path} {option}'[/bold] for help.")

    console.print(
        Panel(
            Text(exc.format_message()),
            title="Error",
            title_align="left",
            border_style=_ERROR_STYLE,
            expand=True,
        )
    )


_original_click_show = click.ClickException.show
_original_usage_show = click.UsageError.show


def _click_show(self: click.ClickException, file: Any = None) -> None:
    if not use_rich_style():
        return _original_click_show(self, file)
    _render_click_exception(self, file)


def _usage_show(self: click.UsageError, file: Any = None) -> None:
    if not use_rich_style():
        return _original_usage_show(self, file)
    _render_click_exception(self, file)


click.ClickException.show = _click_show  # type: ignore[method-assign]
click.UsageError.show = _usage_show  # type: ignore[method-assign]
