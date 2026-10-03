"""Pinned checkpoint merge patch; real tensor/OOM coverage runs on H200."""

import base64
from pathlib import Path

import pytest

from configs.kimi_k3_long_context import build_recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_checkpoint_cpu_merge as patch,
)

SOURCE = Path(__file__).parent / "testdata/miles/k3_checkpoint_utils.py.input"


def test_patch_preserves_surrounding_code_and_is_idempotent(tmp_path):
    target = tmp_path / "utils.py"
    original = SOURCE.read_text()
    target.write_text(original)
    patch.apply(target)
    first = target.read_text()
    assert first.replace(patch.REPLACEMENT, patch.ANCHOR) == original
    assert first == SOURCE.with_suffix(".output").read_text()
    patch.apply(target)
    assert target.read_text() == first


@pytest.mark.parametrize("change", ["missing", "duplicate", "partial", "drift"])
def test_unexpected_source_fails_without_mutation(tmp_path, change):
    source = SOURCE.read_text()
    if change == "missing":
        source = source.replace(patch.ANCHOR, "")
    elif change == "duplicate":
        source += source
    elif change == "partial":
        source += "\n# " + patch.MARKER
    else:
        source = source.replace("if get_args().low_memory_resume:", "if False:")
    target = tmp_path / "utils.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source


def test_only_h200_encodes_checkpoint_patch_and_keeps_optimizer_flags():
    base, h200 = build_recipe(), build_recipe(gpu_type="H200")
    encoded = base64.b64encode(Path(patch.__file__).read_bytes()).decode()
    assert not any(encoded in command for command in base.image_run_commands)
    assert sum(encoded in command for command in h200.image_run_commands) == 1
    assert not h200.extra_config.get("low_memory_resume", False)
    assert h200.optimizer_cpu_offload
    assert h200.optimizer_offload_fraction == 1.0
