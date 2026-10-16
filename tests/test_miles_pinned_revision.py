"""Exercise startup lookups against a real offline Hub cache without refs/main."""

import ast
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest

from modal_dojo.common.launcher_utils import model_revision, resolve_checkpoint_ref
from modal_dojo.frameworks.miles import launcher


@pytest.mark.parametrize("mounted_path", [False, True])
def test_training_startup_reads_pinned_snapshot_without_main(
    tmp_path, monkeypatch, mounted_path
):
    revision = "a" * 40
    snapshot = tmp_path / "models--org--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    # This reproduces the failed startup: the pinned download created no main ref.
    with pytest.raises(LocalEntryNotFoundError):
        snapshot_download("org/model", local_files_only=True)
    model = SimpleNamespace(
        model_name="org/model",
        model_path=str(snapshot) if mounted_path else None,
        revision=revision,
    )
    warmed = []
    namespace = {
        "model": model,
        "miles": SimpleNamespace(model_name="muse_spark"),
        "environment": {},
        "resolve_checkpoint_ref": resolve_checkpoint_ref,
        "model_revision": model_revision,
        "prewarm_remote_code": lambda path, *args, **kwargs: warmed.append(path),
        "snapshot_download": snapshot_download,
    }
    # Execute the actual nested training startup blocks without allocating GPUs.
    tree = ast.parse(Path(launcher.__file__).read_text())
    prewarm = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name)
            and child.value.func.id == "prewarm_remote_code"
            for child in node.body
        )
    )
    exec(
        compile(
            ast.Module(body=[prewarm], type_ignores=[]), "<training-prewarm>", "exec"
        ),
        namespace,
    )
    assert warmed == [str(snapshot)]
    bridge = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "hf_ref"
            for target in node.targets
        )
        and isinstance(node.value, ast.IfExp)
    )
    exec(
        compile(
            ast.Module(body=[bridge], type_ignores=[]), "<bridge-checkpoint>", "exec"
        ),
        namespace,
    )
    assert namespace["hf_ref"] == str(snapshot)
