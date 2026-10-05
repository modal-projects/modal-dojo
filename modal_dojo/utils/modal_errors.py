"""Transient Modal API error classification and bounded-retry helpers.

A single tuple + two runners so every callsite (polling, metadata I/O, app
lifecycle) classifies and retries transport hiccups identically instead of
each deciding which ``modal.exception`` classes are worth another attempt.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from modal.exception import (
    ConnectionError as ModalConnectionError,
    InternalError,
    ResourceExhaustedError,
    ServiceError,
)

TRANSIENT_MODAL_ERRORS: tuple[type[Exception], ...] = (
    ModalConnectionError,
    InternalError,
    ResourceExhaustedError,
    ServiceError,
)

T = TypeVar("T")

_TRANSIENT_ATTEMPTS = 3


def retry_transient(fn: Callable[[], T], *, attempts: int = _TRANSIENT_ATTEMPTS) -> T:
    """Run ``fn``, retrying transient Modal errors with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except TRANSIENT_MODAL_ERRORS:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


async def aretry_transient(
    fn: Callable[[], Awaitable[T]], *, attempts: int = _TRANSIENT_ATTEMPTS
) -> T:
    """``retry_transient`` for async callables."""
    for attempt in range(attempts):
        try:
            return await fn()
        except TRANSIENT_MODAL_ERRORS:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")
