"""Stable, machine-readable errors for the training-gym CLI."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import click


class ExitCode(IntEnum):
    """Process exit codes shared by every training-gym subcommand."""

    SUCCESS = 0
    ERROR = 1  # An unexpected or otherwise unclassified failure occurred.
    USAGE = 2  # The invocation was invalid, such as an unknown flag or missing ID.
    NOT_FOUND = 3  # The requested training-gym resource does not exist.
    AUTH = 4  # Credentials are missing, invalid, or lack permission.
    BACKEND = 5  # The dashboard or Modal is unavailable or returned a bad response.


class CLIError(click.ClickException):
    """An expected CLI failure with agent-readable metadata."""

    def __init__(
        self,
        message: str,
        *,
        error: str,
        exit_code: ExitCode = ExitCode.ERROR,
        hint: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.exit_code = int(exit_code)
        self.hint = hint
        self.details = details

    def format_message(self) -> str:
        message = super().format_message()
        return f"{message}\nHint: {self.hint}" if self.hint else message

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "error": self.error,
            **self.details,
            "message": super().format_message(),
        }
        if self.hint:
            payload["hint"] = self.hint
        return payload
