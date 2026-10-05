"""Extract the first meaningful failure from a distributed job's log stream.

The Ray job's recorded driver message is often a cascade artifact — "rank
exited code 1", "actor is unavailable", "2 groups failed" — while the actual
cause lives in an interleaved worker traceback streamed earlier. This module
re-scans the streamed log for the first fatal signature so the run record can
store the cause instead of the cleanup noise.
"""

from __future__ import annotations

import re

_MAX_LINES = 4
_MAX_CHARS = 2000
# Bounded lookahead for an in-progress traceback block. Tracebacks longer than
# this commit their header line only.
_MAX_TRACEBACK_LINES = 400
# Foreign log lines tolerated between a "During handling" chain marker and the
# chained traceback that follows it (other ranks' output interleaves freely).
_MAX_INTERLEAVE = 8

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
# The exception line closing a traceback block, e.g. "torch.OutOfMemoryError:
# CUDA out of memory." or "RuntimeError: Step 1: 2 groups failed".
_EXCEPTION_LINE_RE = re.compile(
    r"^(?:[\w.]+)?\w*(?:Error|Exception|Interrupt|Exit|Timeout|Aborted|Failure|"
    r"Killed|OOM)\w*:.*"
)

# Fatal signatures worth excerpting when no traceback is present. Ordered
# loosely by diagnostic value; the earliest match in the stream wins.
_SIGNATURE_RES = [
    re.compile(r"Watchdog caught collective operation timeout"),
    re.compile(r"(?:torch\.)?OutOfMemoryError|CUDA out of memory"),
    re.compile(r"\bNCCL\b.*\b(?:error|timeout|unhandled)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:AssertionError|RuntimeError|ValueError|KeyError|TypeError|"
        r"ModuleNotFoundError|ImportError|DeserializationError|RayActorError|"
        r"ActorDiedError|WorkerCrashedError)\b"
    ),
    re.compile(
        r"exited? with (?:exit )?code (?!0\b)\d+|received SIG(?:KILL|TERM|SEGV)"
    ),
]


def _traceback_excerpt(lines: list[str], start: int) -> tuple[str | None, int]:
    """The exception line of the traceback block beginning at ``start``.

    Scans forward until a non-continuation line ends the block; the last
    ``...Error: ...`` line inside it is the cause. Chained exceptions
    ("During handling...", "The above exception...") keep the scan going so
    the terminal exception wins. Returns the exception line and the index of
    the first line past the block, so callers can skip its contents during
    signature matching.
    """
    exception_line: str | None = None
    expect_chain = False
    in_block = True
    interleave = 0
    end = start + 1
    for end in range(start + 1, len(lines)):
        line = lines[end]
        stripped = line.strip()
        if not stripped:
            # Blank lines separate a traceback from its chained continuation.
            continue
        if in_block:
            if stripped.startswith("File ") or line.startswith((" ", "\t")):
                continue
            if _EXCEPTION_LINE_RE.match(stripped):
                exception_line = stripped
                in_block = False
                continue
            if stripped.startswith(("During handling", "The above exception")):
                expect_chain = True
                continue
            break
        # Block is closed by its exception line; only a chained continuation
        # can extend it — unrelated Error-looking log lines must not override.
        if stripped.startswith(("During handling", "The above exception")):
            expect_chain = True
            continue
        if expect_chain:
            if _TRACEBACK_RE.search(stripped):
                expect_chain = False
                in_block = True
                interleave = 0
                continue
            # Another rank's line interleaved between the chain marker and the
            # chained traceback; tolerate a few before giving up on the chain.
            if interleave < _MAX_INTERLEAVE:
                interleave += 1
                continue
        break
    else:
        end = len(lines)
    return exception_line, end


class FailureExcerpt:
    """Collects the first fatal signatures from a streamed log, line by line.

    Memory is bounded: it keeps only the collected excerpt lines plus one
    in-progress traceback block (needed to reach its exception line), so a
    failure early in an arbitrarily long log is still attributed. Feed each
    line via :meth:`feed`; call :meth:`result` when the stream ends.
    """

    def __init__(self) -> None:
        self._excerpt: list[str] = []
        self._pending: list[str] | None = None  # in-progress traceback block
        self._done = False

    def feed(self, line: str) -> None:
        if self._done:
            return
        if self._pending is not None:
            self._pending.append(line)
            if len(self._pending) > _MAX_TRACEBACK_LINES:
                overflow = self._pending[-1]
                self._commit(self._pending[0].strip())
                self._pending = None
                self.feed(overflow)
                return
            excerpt, end = _traceback_excerpt(self._pending, 0)
            if end == len(self._pending):
                # Block still open.
                return
            block, self._pending = self._pending, None
            self._commit(excerpt or block[0].strip())
            # Lines buffered past the block's end may carry signatures.
            for extra in block[end:]:
                self.feed(extra)
            return
        stripped = line.strip()
        if _TRACEBACK_RE.search(stripped):
            self._pending = [line]
            return
        for pattern in _SIGNATURE_RES:
            if pattern.search(stripped):
                self._commit(stripped)
                return

    def _commit(self, text: str) -> None:
        if self._excerpt and self._excerpt[-1] == text:
            return
        self._excerpt.append(text)
        if len(self._excerpt) >= _MAX_LINES:
            self._done = True

    def result(self) -> str | None:
        if self._pending is not None:
            excerpt, _ = _traceback_excerpt(self._pending, 0)
            self._commit(excerpt or self._pending[0].strip())
            self._pending = None
        if not self._excerpt:
            return None
        text = "\n".join(self._excerpt)
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "..."
        return text


def extract_failure_excerpt(lines: list[str]) -> str | None:
    """Return a compact excerpt naming the first fatal event in ``lines``.

    Prefers the exception line of the first Python traceback in the stream;
    falls back to the earliest lines matching a fatal signature (NCCL watchdog
    timeouts, OOMs, actor deaths, non-zero rank exits). Returns ``None`` when
    nothing in the stream looks like a failure — callers should keep their
    existing message in that case.
    """
    collector = FailureExcerpt()
    for line in lines:
        collector.feed(line)
    return collector.result()
