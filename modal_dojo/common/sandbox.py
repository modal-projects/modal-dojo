from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import modal

_DEFAULT_APP_NAME = "training-gym-sandbox"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a single ``Sandbox.run`` invocation."""

    stdout: str
    stderr: str
    returncode: int


class Sandbox:
    """Controls a [Modal Sandbox](https://modal.com/docs/guide/sandbox) in the ``app_name`` app."""

    def __init__(
        self, *, app_name: str = _DEFAULT_APP_NAME, **create_kwargs: Any
    ) -> None:
        self._app_name = app_name
        self._create_kwargs = create_kwargs
        self._sb: modal.Sandbox | None = None

    def __enter__(self) -> Sandbox:
        self._sb = modal.Sandbox.create(
            "sleep",
            "infinity",
            app=modal.App.lookup(self._app_name, create_if_missing=True),
            **self._create_kwargs,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        sb = self._sb
        if sb is None:
            return
        self._sb = None
        try:
            sb.terminate()
        except Exception:
            logger.exception(
                "Sandbox %s failed to terminate; it runs until its timeout",
                sb.object_id,
            )
        sb.detach()

    def write(self, path: str, data: str | bytes) -> None:
        if isinstance(data, str):
            self._sb.filesystem.write_text(data, path)
        else:
            self._sb.filesystem.write_bytes(data, path)

    def run(self, *cmd: str, timeout: int | None = None) -> SandboxResult:
        proc = self._sb.exec(*cmd, timeout=timeout)
        proc.wait()
        return SandboxResult(
            stdout=proc.stdout.read(),
            stderr=proc.stderr.read(),
            returncode=proc.returncode,
        )


def open_sandbox(
    *,
    app_name: str = _DEFAULT_APP_NAME,
    setup: Sequence[Sequence[str]] = (),
    **create_kwargs: Any,
) -> Sandbox:
    """A started ``Sandbox``; close it with ``__exit__``.

    ``create_kwargs`` set to ``None`` are left out, so callers can pass optional
    resources (``cpu``, ``memory``, ``workdir``) straight through. Each ``setup``
    command runs before returning; if one raises, the sandbox is closed.
    """
    kwargs = {k: v for k, v in create_kwargs.items() if v is not None}
    sb = Sandbox(app_name=app_name, **kwargs).__enter__()
    try:
        for cmd in setup:
            sb.run(*cmd)
    except BaseException:
        sb.__exit__(None, None, None)
        raise
    return sb
