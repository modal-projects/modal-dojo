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
# Foreign log lines tolerated inside a traceback block, or between a
# "During handling" chain marker and the chained traceback that follows it
# (other ranks' output interleaves freely).
_MAX_INTERLEAVE = 8

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
# Ray prefixes worker output per line ("(TrainActor pid=N) ...",
# "[rank N] ..."); strip one leading "(...)" or "[...]" group so prefixed
# frames, source lines, and exception lines classify like unprefixed ones.
_RAY_PREFIX_RE = re.compile(r"^\([^()\n]*\)\s*|^\[[^\[\]\n]*\d[^\[\]\n]*\]\s*")
_LINE_SPLIT_RE = re.compile(r"\r\n|\r|\n")
_MAX_TAIL_CHARS = 8 * 1024
# Literal spans that make up signatures — a truncation cut inside one keeps
# its halves from ever matching, so the cut must land outside them.
_SIGNATURE_LITERALS = (
    "Watchdog caught collective operation timeout",
    "Traceback (most recent call last)",
    "exited with code",
    "exit with code",
    "received SIGKILL",
    "received SIGTERM",
    "received SIGSEGV",
    "OutOfMemoryError",
    "CUDA out of memory",
    "NCCL",
)


def _live_prefix_len(text: str) -> int:
    best = 0
    for literal in _SIGNATURE_LITERALS:
        for i in range(min(len(literal) - 1, len(text)), best, -1):
            if text.endswith(literal[:i]):
                best = i
                break
    return best


# The exception line closing a traceback block, e.g. "torch.OutOfMemoryError:
# CUDA out of memory." or "RuntimeError: Step 1: 2 groups failed". The
# ": message" part is optional — bare raises (KeyboardInterrupt, SystemExit)
# print the exception name alone.
_EXCEPTION_LINE_RE = re.compile(
    r"^[\w.]*?(?:Error|Exception|Interrupt|Exit|Timeout|Aborted|Failure|"
    r"Killed|OOM)[\w.]*(?::.*)?$"
)
# The same exception-name shape appearing mid-line — Ray prefixes worker
# output per line ("(TrainActor pid=N) ...", "[rank N] ..."), so the closing
# line of a worker traceback may not start at column 0.
_PREFIXED_EXCEPTION_RE = re.compile(
    r"\b\w+(?:Error|Exception|Interrupt|Exit|Timeout|Aborted|Failure|Killed|"
    r"OOM)\w*:"
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
    # Any exception line stranded outside a parsed traceback — a block
    # abandoned to interleaving, or a custom error name not listed above.
    _PREFIXED_EXCEPTION_RE,
    _EXCEPTION_LINE_RE,
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
        line = _RAY_PREFIX_RE.sub("", lines[end])
        stripped = line.strip()
        if not stripped:
            # Blank lines separate a traceback from its chained continuation.
            continue
        if in_block:
            if stripped.startswith("File ") or line.startswith((" ", "\t")):
                continue
            if _EXCEPTION_LINE_RE.match(stripped) or _PREFIXED_EXCEPTION_RE.search(
                stripped
            ):
                exception_line = stripped
                in_block = False
                continue
            if stripped.startswith(("During handling", "The above exception")):
                expect_chain = True
                continue
            # Foreign line mid-block — another rank's output interleaved into
            # the traceback, or a per-line prefix on a frame. Tolerate a few
            # before giving up on reaching the exception line.
            if interleave < _MAX_INTERLEAVE:
                interleave += 1
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

    Memory is bounded: it keeps only the collected excerpt lines, one
    in-progress traceback block (needed to reach its exception line), and an
    unterminated line tail, so a failure early in an arbitrarily long log is
    still attributed. Feed raw chunks via :meth:`feed_chunk`; call
    :meth:`result` when the stream ends.
    """

    def __init__(self) -> None:
        self._excerpt: list[str] = []
        self._pending: list[str] | None = None  # in-progress traceback block
        self._buf = ""  # unterminated line tail carried across chunks
        self._done = False

    def feed_chunk(self, text: str) -> None:
        """Feed a raw stream chunk.

        Only newline-terminated lines are consumed; the unterminated tail is
        buffered onto the next chunk so a signature split across the stream's
        boundaries is still seen whole.
        """
        if self._done:
            return
        lines = _LINE_SPLIT_RE.split(self._buf + text)
        self._buf = lines.pop()
        for line in lines:
            self.feed(line)
        if len(self._buf) > _MAX_TAIL_CHARS:
            head = self._buf[:-_MAX_TAIL_CHARS]
            boundary = 0
            for match in re.finditer(r"[^\w.]", head):
                candidate = match.end()
                if _live_prefix_len(head[:candidate]) == 0:
                    boundary = candidate
            if not boundary:
                boundary = len(head) - _live_prefix_len(head)
            boundary = max(boundary, len(head) - _MAX_TAIL_CHARS)
            self.feed(head[:boundary])
            self._buf = head[boundary:] + self._buf[-_MAX_TAIL_CHARS:]

    def feed(self, line: str) -> None:
        if self._done:
            return
        if self._pending is not None:
            self._pending.append(line)
            if len(self._pending) > _MAX_TRACEBACK_LINES:
                overflow = self._pending[-1]
                self._commit(_RAY_PREFIX_RE.sub("", self._pending[0]).strip())
                self._pending = None
                self.feed(overflow)
                return
            excerpt, end = _traceback_excerpt(self._pending, 0)
            if end == len(self._pending):
                # Block still open.
                return
            block, self._pending = self._pending, None
            self._commit(excerpt or _RAY_PREFIX_RE.sub("", block[0]).strip())
            # Lines buffered past the block's end may carry signatures.
            for extra in block[end:]:
                self.feed(extra)
            return
        stripped = _RAY_PREFIX_RE.sub("", line).strip()
        if _TRACEBACK_RE.search(stripped):
            self._pending = [line]
            return
        for pattern in _SIGNATURE_RES:
            match = pattern.search(stripped)
            if match:
                start = max(0, match.end() - _MAX_CHARS)
                self._commit(stripped[start : start + _MAX_CHARS])
                return

    def flush(self) -> None:
        """Terminate the buffered partial line (e.g. a stream ended mid-line)."""
        if self._buf:
            self.feed(self._buf)
            self._buf = ""

    def _commit(self, text: str) -> None:
        # Global dedup: a reconnected log stream can replay earlier lines, and
        # several ranks often emit the identical signature.
        if text in self._excerpt:
            return
        self._excerpt.append(text)
        if len(self._excerpt) >= _MAX_LINES:
            self._done = True

    def result(self) -> str | None:
        if self._buf:
            self.feed(self._buf)
            self._buf = ""
        if self._pending is not None:
            excerpt, _ = _traceback_excerpt(self._pending, 0)
            self._commit(excerpt or _RAY_PREFIX_RE.sub("", self._pending[0]).strip())
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
