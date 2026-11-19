from __future__ import annotations

import io
import logging
import shlex
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

_DEFAULT_APP_NAME = "training-gym-sandbox"
_TRANSFER_ARCHIVE = "/tmp/.modal-dojo-transfer.tar.gz"

logger = logging.getLogger(__name__)


def tar_gz(directory: Path, *, exclude: Iterable[str] = ()) -> bytes:
    """Gzipped tarball of ``directory``'s contents, minus top-level ``exclude`` names."""
    skip = set(exclude)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(directory.iterdir()):
            if child.name not in skip:
                tar.add(child, arcname=child.name)
    return buf.getvalue()


def extract_tar_gz(data: bytes, dest: Path) -> None:
    """Unpack a ``tar_gz`` archive into ``dest``, refusing unsafe members."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


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

    def read(self, path: str) -> bytes:
        return self._sb.filesystem.read_bytes(path)

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        """Copy ``local_dir``'s contents into ``remote_dir``, creating it."""
        self.write(_TRANSFER_ARCHIVE, tar_gz(local_dir))
        target = shlex.quote(remote_dir)
        self._check(
            f"mkdir -p {target} && tar -xzf {_TRANSFER_ARCHIVE} -C {target}"
            f" && rm -f {_TRANSFER_ARCHIVE}"
        )

    def download_dir(self, remote_dir: str, local_dir: Path) -> None:
        """Copy ``remote_dir``'s contents into ``local_dir``, creating it."""
        self._check(f"tar -czf {_TRANSFER_ARCHIVE} -C {shlex.quote(remote_dir)} .")
        extract_tar_gz(self.read(_TRANSFER_ARCHIVE), local_dir)

    def _check(self, script: str) -> None:
        result = self.run("bash", "-c", script)
        if result.returncode != 0:
            raise RuntimeError(
                f"Sandbox command failed ({result.returncode}): {script}\n"
                f"{result.stdout}{result.stderr}"
            )

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
