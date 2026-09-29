"""CLI entry point: ``training-gym <command>``."""

from __future__ import annotations

import sys

import click

from .builtin import (
    cleanup_command,
    open_command,
    set_password_command,
    set_proxy_auth_command,
    setup_command,
)
from .commands import _TrainingGymGroup
from .errors import CLIError, ExitCode
from .output import print_error, print_json
from .run import run_group
from .skills import skills_group


@click.group(
    cls=_TrainingGymGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
def entrypoint_cli() -> None:
    """Manage training runs, skills, and dashboard access."""


def _register_commands() -> None:
    entrypoint_cli.add_command(run_group, panel="Training runs")
    entrypoint_cli.add_command(skills_group, panel="Skills")
    entrypoint_cli.add_command(setup_command, panel="Configuration")
    entrypoint_cli.add_command(set_password_command, panel="Configuration")
    entrypoint_cli.add_command(set_proxy_auth_command, panel="Configuration")
    entrypoint_cli.add_command(open_command, panel="Utilities")
    entrypoint_cli.add_command(cleanup_command, panel="Utilities")


_register_commands()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = list(argv) if argv is not None else sys.argv[1:]
    json_output = "-j" in args or "--json" in args
    try:
        result = entrypoint_cli.main(
            args=args,
            prog_name="training-gym",
            standalone_mode=False,
        )
        return int(result) if isinstance(result, int) else int(ExitCode.SUCCESS)
    except CLIError as exc:
        if json_output:
            print_json(exc.to_json(), stderr=True)
        else:
            exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.ClickException as exc:
        if json_output:
            print_json(
                {
                    "error": (
                        "usage_error"
                        if isinstance(exc, click.UsageError)
                        else "command_error"
                    ),
                    "message": exc.format_message(),
                },
                stderr=True,
            )
        else:
            exc.show(file=sys.stderr)
        return int(exc.exit_code)
    except click.Abort:
        if json_output:
            print_json({"error": "interrupted"}, stderr=True)
        else:
            print_error("Interrupted.")
        return 130
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        if json_output:
            print_json(
                {"error": "unexpected_error", "message": message},
                stderr=True,
            )
        else:
            print_error(f"Error: {message}")
        return int(ExitCode.ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
