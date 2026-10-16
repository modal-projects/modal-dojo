"""Run the actual nested conversion function with its GPU process replaced."""

import ast
import contextlib
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from modal_dojo.common.torch_dist_checkpoint import (
    is_complete_torch_dist_checkpoint_storage,
)
from modal_dojo.frameworks.miles import launcher


@pytest.mark.parametrize("complete", [False, True])
def test_conversion_publishes_and_reloads_before_verifying(complete):
    tree = ast.parse(Path(launcher.__file__).read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "convert_checkpoint"
    )
    function.decorator_list = []
    events = []

    def run(command, *, check, env):
        assert check
        assert env["MODAL_DOJO_CHECKPOINTS_VOLUME_NAME"] == "shared-checkpoints"
        events.append("convert")

    def verify(path, *, is_complete):
        assert is_complete is is_complete_torch_dist_checkpoint_storage
        events.append("verify")
        return complete

    namespace = {
        "miles": SimpleNamespace(
            ref_load="/checkpoints/model",
            miles_model_script=None,
            model_name="muse_spark",
        ),
        "model": object(),
        "environment": {},
        "os": os,
        "shlex": shlex,
        "contextlib": contextlib,
        "MILES_ROOT": "/root/miles",
        "get_checkpoint_conversion_policy": lambda *args, **kwargs: (1, 8, []),
        "model_args_command": lambda *args: None,
        "checkpoints_volume_name": "shared-checkpoints",
        "_refresh_convert_lock": lambda *args: None,
        "_convert_lock_heartbeat": lambda *args: contextlib.nullcontext(),
        "_release_convert_lock": lambda *args: events.append("release"),
        "prewarm_remote_code": lambda *args, **kwargs: None,
        "subprocess": SimpleNamespace(run=run),
        "checkpoints_volume": SimpleNamespace(
            commit=lambda: events.append("commit"),
            reload=lambda: events.append("reload"),
        ),
        "has_torch_dist_checkpoint": verify,
        "is_complete_torch_dist_checkpoint_storage": is_complete_torch_dist_checkpoint_storage,
        "flush_status_reporter": lambda **kwargs: None,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<conversion>", "exec"),
        namespace,
    )
    if complete:
        namespace["convert_checkpoint"]("/cached/model")
    else:
        with pytest.raises(RuntimeError, match="holds no complete"):
            namespace["convert_checkpoint"]("/cached/model")
    assert events == ["convert", "commit", "reload", "verify", "release"]
