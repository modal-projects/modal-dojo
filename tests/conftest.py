"""Shared pytest fixtures and test doubles."""

from __future__ import annotations

import io

import pytest

from modal_dojo.utils import metadata


class _Method:
    """A Modal Volume method exposing a sync call and an ``.aio`` async form."""

    def __init__(self, sync_fn, async_fn):
        self._sync_fn = sync_fn
        self.aio = async_fn

    def __call__(self, *args, **kwargs):
        return self._sync_fn(*args, **kwargs)


class FakeVolume:
    """In-memory stand-in for a Modal Volume that is *not* attached.

    ``reload()`` raises like a real unattached/local volume (both sync and via
    ``.aio()``); reads and writes operate on an in-memory dict. A correct
    metadata layer must still complete a ``save()`` (sync or ``is_async=True``)
    against this — reload is only a freshness hint.
    """

    class _DirEntry:
        """Simple stand-in for Modal Volume directory entries."""

        def __init__(self, path: str, *, mtime: int, size: int):
            self.path = path
            self.mtime = mtime
            self.size = size

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.reload = _Method(self._reload, self._reload_async)
        self.read_file = _Method(self._read_file, self._read_file_async)
        self.iterdir = _Method(self._iterdir, self._iterdir_async)

    def _reload(self) -> None:
        raise RuntimeError("reload() can only be called from within a running function")

    async def _reload_async(self) -> None:
        raise RuntimeError("reload() can only be called from within a running function")

    def _read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    async def _read_file_async(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def remove_file(self, path: str) -> None:
        if path not in self.files:
            raise FileNotFoundError(path)
        del self.files[path]

    def _iterdir(self, path: str):
        prefix = path.rstrip("/") + "/"
        return [
            self._DirEntry(f, mtime=1, size=len(self.files[f]))
            for f in self.files
            if f.startswith(prefix)
        ]

    async def _iterdir_async(self, path: str):
        for entry in self._iterdir(path):
            yield entry

    def batch_upload(self, force: bool = False):
        files = self.files

        class _Batch:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def put_file(self, fileobj: io.BytesIO, path: str) -> None:
                files[path] = fileobj.read()

        return _Batch()


@pytest.fixture
def fake_volume(monkeypatch) -> FakeVolume:
    """Swap the metadata volume for an in-memory ``FakeVolume`` (no Modal, no GPU)."""
    vol = FakeVolume()
    monkeypatch.setattr(metadata, "_metadata_volume", lambda: vol)
    return vol


def pytest_addoption(parser):
    parser.addoption(
        "--rewrite",
        action="store_true",
        default=False,
        help="Rewrite golden .output files in tests/testdata/ instead of asserting",
    )
