"""Diagnostics must preserve execution and capture stalls without CUDA."""

import json
import builtins
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_training_gym.frameworks.miles import glm_5_3_flash_lora_debug as debug
from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_lora_debug as patch,
)


def test_pinned_actor_patch_contract():
    path = Path(__file__).parent / "testdata/miles/glm_5_3_flash_lora/actor.py.output"
    source = path.read_text()
    result = patch.patch_source(source)
    assert result == source + patch.SUFFIX
    assert patch.patch_source(result) == result
    compile(result, "actor.py", "exec")
    with pytest.raises(ValueError, match="missing GLM diagnostic methods"):
        patch.patch_source(source.replace("def compute_log_prob(", "def changed_api("))


def test_debug_disabled_does_not_touch_actor(monkeypatch):
    monkeypatch.delenv(debug.ENABLE_ENV, raising=False)

    class Actor:
        pass

    debug.instrument_actor(Actor)
    assert not hasattr(Actor, "_glm53_debug_wrapped")


def test_enabled_actor_preserves_method_binding(tmp_path, monkeypatch):
    monkeypatch.setenv(debug.ENABLE_ENV, "1")
    monkeypatch.setenv("GLM53_DEBUG_DIR", str(tmp_path))
    state = debug.Diagnostics(8, start_observer=False)
    monkeypatch.setattr(debug, "install", lambda rank: state)

    class Actor:
        _rank = 8

        def compute_log_prob(self, value, *, offset=0):
            return self._rank + value + offset

        init = wake_up = train_actor = update_weights = compute_log_prob

    debug.instrument_actor(Actor)
    wrapped = Actor.compute_log_prob
    debug.instrument_actor(Actor)
    assert Actor.compute_log_prob is wrapped
    assert Actor().compute_log_prob(3, offset=2) == 13
    assert json.loads(state.path.read_text())["stack"] == []


def test_kernel_hook_skips_loaded_modules(monkeypatch):
    class Kernel:
        module = None
        name = "test_sort"

        def _init_handles(self):
            self.module = 123
            return "original_result"

    state = Mock()
    dispatcher = SimpleNamespace(sort_chunks_by_idxs=lambda x: x)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton.compiler.compiler":
            return SimpleNamespace(CompiledKernel=Kernel)
        if name == "megatron.core.transformer.moe":
            return SimpleNamespace(token_dispatcher=dispatcher)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(debug, "_state", None)
    monkeypatch.setattr(debug, "Diagnostics", lambda rank: state)
    debug.install(8)
    kernel = Kernel()
    assert kernel._init_handles() == "original_result"
    assert kernel._init_handles() == "original_result"
    state.enter.assert_called_once_with("triton_load:test_sort")
    assert state.leave.call_count == 1


def test_nested_spans_preserve_results_and_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("GLM53_DEBUG_DIR", str(tmp_path))
    state = debug.Diagnostics(8, start_observer=False)
    outer = state.enter("train_actor")
    assert debug._wrap(lambda x: x + 1, state, "moe_sort_chunks")(2) == 3

    def fail():
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        debug._wrap(fail, state, "triton_load:test")()
    assert json.loads(state.path.read_text())["stack"] == [outer]
    state.leave(outer)
    assert json.loads(state.path.read_text())["stack"] == []


def test_observer_reports_simulated_stall(tmp_path):
    # Stand-ins verify observer invocation without attaching to a GPU/process.
    for name in ("py-spy", "nvidia-smi"):
        executable = tmp_path / name
        executable.write_text("#!/bin/sh\nprintf 'DIAGNOSTIC_STUB_OK\\n'\n")
        executable.chmod(0o755)
    source = """
import time
import faulthandler
from modal_training_gym.frameworks.miles.glm_5_3_flash_lora_debug import Diagnostics
def reject_in_process_dump(*args, **kwargs):
    raise AssertionError("Trainer must not schedule in-process traceback dumps")
faulthandler.dump_traceback_later = reject_in_process_dump
d = Diagnostics(8)
t = d.enter("triton_load:simulated_block")
try:
    time.sleep(16)
finally:
    d.leave(t)
    d.observer.terminate()
    d.observer.wait(timeout=5)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        text=True,
        capture_output=True,
        timeout=25,
        env={
            **os.environ,
            "GLM53_DEBUG_DIR": str(tmp_path),
            "GLM53_DEBUG_STALL_SECONDS": "10",
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        },
    )
    assert result.returncode == 0, result.stderr
    assert "observer_stall" in result.stdout
    assert "triton_load:simulated_block" in result.stdout
    assert result.stdout.count("DIAGNOSTIC_STUB_OK") == 2
    assert "Timeout" not in result.stderr
