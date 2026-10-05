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
    re.compile(r"exited? with (?:exit )?code \d+|received SIG(?:KILL|TERM|SEGV)"),
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
        if expect_chain and _TRACEBACK_RE.search(stripped):
            expect_chain = False
            in_block = True
            continue
        break
    else:
        end = len(lines)
    return exception_line, end


def extract_failure_excerpt(lines: list[str]) -> str | None:
    """Return a compact excerpt naming the first fatal event in ``lines``.

    Prefers the exception line of the first Python traceback in the stream;
    falls back to the earliest line matching a fatal signature (NCCL watchdog
    timeouts, OOMs, actor deaths, non-zero rank exits). Returns ``None`` when
    nothing in the stream looks like a failure — callers should keep their
    existing message in that case.
    """
    excerpt: list[str] = []
    skip_until = -1
    for i, line in enumerate(lines):
        if i < skip_until:
            continue
        stripped = line.strip()
        matched: str | None = None
        if _TRACEBACK_RE.search(stripped):
            matched, skip_until = _traceback_excerpt(lines, i)
            if matched is None:
                matched = stripped
        else:
            for pattern in _SIGNATURE_RES:
                if pattern.search(stripped):
                    matched = stripped
                    break
        if matched is not None and (not excerpt or matched != excerpt[-1]):
            excerpt.append(matched)
            if len(excerpt) >= _MAX_LINES:
                break
    if not excerpt:
        return None
    text = "\n".join(excerpt)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "..."
    return text
