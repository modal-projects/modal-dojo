"""Client helpers for the NCCL RAS (reliability/availability/serviceability) subsystem.

Since NCCL 2.24, every NCCL process runs a RAS thread that answers plain-text
queries over a TCP socket (default ``localhost:28028``). Binding it to
``0.0.0.0`` via ``NCCL_RAS_ADDR`` lets the cluster head node query every
node's RAS thread: ``STATUS`` reports job processes it considers dead or
unresponsive, and an unreachable socket signals a stuck node or a networking
problem.
"""

from __future__ import annotations

import socket

NCCL_RAS_PORT = 28028
# Baking this into the training image binds each node's RAS socket to all
# interfaces so the head node can reach it; recipe ``image_env``/``environment``
# entries still override it.
NCCL_RAS_IMAGE_ENV = {"NCCL_RAS_ADDR": f"0.0.0.0:{NCCL_RAS_PORT}"}

_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 15.0

_PROBLEM_TOKENS = ("dead", "unresponsive", "unreachable")


def query_status(
    host: str,
    port: int = NCCL_RAS_PORT,
    *,
    connect_timeout: float = _CONNECT_TIMEOUT_SECONDS,
    read_timeout: float = _READ_TIMEOUT_SECONDS,
) -> str:
    """Return the RAS ``STATUS`` response from ``host:port``."""
    with socket.create_connection((host, port), timeout=connect_timeout) as sock:
        sock.settimeout(read_timeout)
        sock.sendall(b"STATUS\n")
        chunks = []
        while True:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def problem_lines(status: str) -> list[str]:
    """Extract the lines of a ``STATUS`` response that report job problems."""
    return [
        line.strip()
        for line in status.splitlines()
        if any(token in line.lower() for token in _PROBLEM_TOKENS)
    ]
