"""Opt-in, process-local diagnostics for the pinned GLM53 bridge LoRA recipe.

The observer is a fresh CPU-only process: it can report a blocked CUDA driver
call even when that call holds the trainer's Python GIL. No CUDA operations or
collectives are inserted into training. This module imports only the stdlib.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ENABLE_ENV = "TRAINING_GYM_GLM53_LORA_DEBUG"
_state = None


class Diagnostics:
    def __init__(self, rank, *, start_observer=True):
        self.rank = rank
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.stack = []
        self.sequence = 0
        self.counts = {}
        self.interval = max(10, int(os.environ.get("GLM53_DEBUG_STALL_SECONDS", "120")))
        root = Path(os.environ.get("GLM53_DEBUG_DIR", "/tmp/glm53-lora-debug"))
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{socket.gethostname()}-{self.pid}.json"
        self.publish()
        self.observer = None
        if start_observer:
            self.observer = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--watch",
                    str(self.pid),
                    str(self.path),
                    str(self.interval),
                ],
                stdin=subprocess.DEVNULL,
            )
            if sys.platform == "linux":
                # Yama rejects a child attaching to its parent by default.
                # Allow this observer and its descendants, not every process.
                import ctypes

                libc = ctypes.CDLL(None, use_errno=True)
                status = libc.prctl(
                    0x59616D61, self.observer.pid, 0, 0, 0
                )  # PR_SET_PTRACER
                if status != 0:
                    self.emit("native_attach_unavailable", errno=ctypes.get_errno())
        versions = {}
        for package in (
            "torch",
            "triton",
            "transformer_engine",
            "megatron-core",
            "megatron-bridge",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unavailable"
        self.emit("installed", versions=versions, state_file=str(self.path))

    def emit(self, event, **fields):
        print(
            "GLM53_DEBUG "
            + json.dumps(
                {
                    "event": event,
                    "pid": self.pid,
                    "rank": self.rank,
                    "time": time.time(),
                    **fields,
                }
            ),
            flush=True,
        )

    def publish(self):
        self.sequence += 1
        payload = {
            "pid": self.pid,
            "rank": self.rank,
            "host": socket.gethostname(),
            "sequence": self.sequence,
            "updated": time.monotonic(),
            "stack": self.stack,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(self.path)

    def enter(self, label):
        with self.lock:
            self.counts[label] = self.counts.get(label, 0) + 1
            # MoE sorting runs every layer/microbatch. Keep its live state but
            # log only initial calls and periodic progress to bound log volume.
            emit = (
                label != "moe_sort_chunks"
                or self.counts[label] <= 2
                or self.counts[label] % 100 == 0
            )
            token = {
                "label": label,
                "started": time.monotonic(),
                "emit": emit,
                "count": self.counts[label],
            }
            self.stack.append(token)
            self.publish()
            if emit:
                self.emit("begin", label=label, count=token["count"])
            # Keep stack collection outside the trainer. In the pinned runtime,
            # an all-thread faulthandler timer coincided with rank 0 exiting
            # halfway through an interpreter-trampoline frame dump.
            return token

    def leave(self, token, ok=True):
        with self.lock:
            if token["emit"] or not ok:
                self.emit(
                    "end",
                    label=token["label"],
                    ok=ok,
                    elapsed_s=round(time.monotonic() - token["started"], 3),
                )
            self.stack.remove(token)
            self.publish()


def _wrap(function, diagnostics, label):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        token = diagnostics.enter(label)
        ok = False
        try:
            result = function(*args, **kwargs)
            ok = True
            return result
        finally:
            diagnostics.leave(token, ok)

    return wrapped


def install(rank):
    global _state
    if _state is not None:
        return _state
    _state = Diagnostics(rank)
    # Instrument the exact Python boundary above cuModuleLoadData. Module-load
    # calls are infrequent compared with kernel execution; no per-token hook.
    from triton.compiler.compiler import CompiledKernel

    original = CompiledKernel._init_handles

    @functools.wraps(original)
    def load_kernel(kernel, *args, **kwargs):
        # Triton can call this guard on every launch/autotuning iteration.
        # Observe real module loads only, not already initialized kernels.
        if getattr(kernel, "module", None) is not None:
            return original(kernel, *args, **kwargs)
        label = "triton_load:" + str(getattr(kernel, "name", "unknown"))
        return _wrap(original, _state, label)(kernel, *args, **kwargs)

    CompiledKernel._init_handles = load_kernel
    # This alias is what Megatron's all-to-all dispatcher actually calls.
    from megatron.core.transformer.moe import token_dispatcher

    token_dispatcher.sort_chunks_by_idxs = _wrap(
        token_dispatcher.sort_chunks_by_idxs, _state, "moe_sort_chunks"
    )
    return _state


def instrument_actor(cls):
    if os.environ.get(ENABLE_ENV) != "1" or getattr(cls, "_glm53_debug_wrapped", False):
        return
    for name in (
        "init",
        "wake_up",
        "compute_log_prob",
        "train_actor",
        "update_weights",
    ):
        original = getattr(cls, name)

        def make_wrapper(function, label):
            @functools.wraps(function)
            def wrapped(actor, *args, **kwargs):
                diagnostics = install(
                    getattr(actor, "_rank", os.environ.get("RANK", "unknown"))
                )
                return _wrap(function, diagnostics, label)(actor, *args, **kwargs)

            return wrapped

        setattr(cls, name, make_wrapper(original, name))
    cls._glm53_debug_wrapped = True


def watch(pid: int, path: Path, interval: int):
    last_dump = 0.0
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(5)
            continue
        now = time.monotonic()
        if (
            state["stack"]
            and now - state["updated"] >= interval
            and now - last_dump >= interval
        ):
            last_dump = now
            print("GLM53_DEBUG observer_stall " + json.dumps(state), flush=True)
            captures = []
            # No locals or environment dump: only stacks and GPU counters.
            for command in (
                ["py-spy", "dump", "--native", "--pid", str(pid)],
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv",
                ],
            ):
                try:
                    result = subprocess.run(
                        command, capture_output=True, text=True, timeout=20
                    )
                    capture = {
                        "pid": pid,
                        "rank": state["rank"],
                        "command": command,
                        "returncode": result.returncode,
                        "stdout": result.stdout[-24000:],
                        "stderr": result.stderr[-2000:],
                    }
                    captures.append(capture)
                    print(
                        "GLM53_DEBUG observer_command " + json.dumps(capture),
                        flush=True,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    print(f"GLM53_DEBUG observer_error pid={pid}: {error}", flush=True)
            path.with_suffix(".capture.json").write_text(json.dumps(captures))
        time.sleep(5)


if __name__ == "__main__" and sys.argv[1:2] == ["--watch"]:
    watch(int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
