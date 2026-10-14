"""Client helpers for the NCCL RAS (reliability/availability/serviceability) subsystem.

Since NCCL 2.24, every NCCL process runs a RAS thread that answers plain-text
queries over a TCP socket (default ``localhost:28028``). Binding it to
``0.0.0.0`` via ``NCCL_RAS_ADDR`` lets the cluster head node query every
node's RAS thread: ``STATUS`` reports job processes it considers dead or
unresponsive, and an unreachable socket signals a stuck node or a networking
problem.

RAS has no authentication; ``0.0.0.0`` exposes job status on the node's
cluster interface, reachable from sibling containers on the same Modal
cluster network.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Mapping

NCCL_RAS_PORT = 28028
# Baking this into the training image binds each node's RAS socket to all
# interfaces so the head node can reach it; recipe ``image_env``/``environment``
# entries still override it.
NCCL_RAS_IMAGE_ENV = {"NCCL_RAS_ADDR": f"0.0.0.0:{NCCL_RAS_PORT}"}

_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 15.0

_PROBLEM_TOKENS = ("dead", "unresponsive", "unreachable")
_ZERO_COUNT = re.compile(
    r"\b0\s+(?:\w+\s+){0,3}?(?:dead|unresponsive|unreachable)\w*\b", re.IGNORECASE
)


def ras_port_from_env(env: Mapping[str, str] | None = None) -> int:
    """RAS port parsed from ``NCCL_RAS_ADDR`` (``host:port``), else the default."""
    addr = (env or os.environ).get("NCCL_RAS_ADDR", "")
    port = addr.rpartition("]:")[2] if "]:" in addr else addr.rpartition(":")[2]
    try:
        return int(port)
    except ValueError:
        return NCCL_RAS_PORT


def query_status(
    host: str,
    port: int = NCCL_RAS_PORT,
    *,
    connect_timeout: float = _CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = _READ_TIMEOUT_SECONDS,
) -> str:
    """Return the RAS ``STATUS`` response from ``host:port``.

    NCCL normally closes the client connection after replying; if it keeps the
    socket open, a read that already returned data is still treated as a
    complete response rather than an unreachable endpoint.
    """
    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        sock.settimeout(read_timeout)
        sock.sendall(b"STATUS\n")
        chunks = []
        try:
            while True:
                chunk = sock.recv(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout:
            if not chunks:
                raise
    return b"".join(chunks).decode("utf-8", errors="replace")


def problem_lines(status: str) -> list[str]:
    """Extract the lines of a ``STATUS`` response that report job problems."""
    lines = []
    for line in status.splitlines():
        stripped = line.strip()
        # Healthy summaries can name the tokens as zero counts, e.g.
        # "8 processes, 0 dead"; remove those before matching.
        candidate = _ZERO_COUNT.sub("", stripped)
        if any(token in candidate.lower() for token in _PROBLEM_TOKENS):
            lines.append(stripped)
    return lines
