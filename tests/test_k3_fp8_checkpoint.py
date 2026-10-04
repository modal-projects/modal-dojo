"""Pinned-source guards; real load round trips and OOM reproduction run on H200."""

import base64
from pathlib import Path

import pytest

from configs.kimi_k3_h200 import build_recipe
from modal_dojo import Kimi_K3_LoRA_Recipe
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_fp8_checkpoint as patch,
)
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_fused_lora as fusion,
)

DATA = Path(__file__).parent / "testdata/miles"


def originals(tmp_path):
    checkpoint, lora = tmp_path / "serialization.py", tmp_path / "lora.py"
    checkpoint.write_text((DATA / "k3_checkpoint_serialization.py.input").read_text())
    lora.write_text((DATA / "k3_lora.py.input").read_text())
    return checkpoint, lora


def test_load_only_patch_preserves_save_and_composes_with_lora_fusion(tmp_path):
    checkpoint, lora = originals(tmp_path)
    original = checkpoint.read_text()
    fusion.apply(lora)
    fused = lora.read_text()
    patch.apply(checkpoint, lora)
    patched = checkpoint.read_text(), lora.read_text()
    assert patched[0].replace(patch.LOAD_REPLACEMENT, patch.LOAD_ANCHOR) == original
    assert patched[1].replace(patch.LORA_REPLACEMENT, patch.LORA_ANCHOR) == fused
    assert patched[0].split("\ndef save(", 1)[1] == original.split("\ndef save(", 1)[1]
    patch.apply(checkpoint, lora)
    assert patched == (checkpoint.read_text(), lora.read_text())


@pytest.mark.parametrize("file_index", [0, 1])
@pytest.mark.parametrize("drift", ["missing", "duplicate", "partial"])
def test_both_files_validated_before_either_is_changed(tmp_path, file_index, drift):
    paths = originals(tmp_path)
    anchor, marker = (
        (patch.LOAD_ANCHOR, patch.LOAD_MARKER),
        (patch.LORA_ANCHOR, patch.LORA_MARKER),
    )[file_index]
    path = paths[file_index]
    source = path.read_text()
    if drift == "missing":
        source = source.replace(anchor, "")
    elif drift == "duplicate":
        source += anchor
    else:
        source += "\n# " + marker
    path.write_text(source)
    before = [p.read_text() for p in paths]
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(*paths)
    assert [p.read_text() for p in paths] == before


def test_only_h200_adds_fp8_checkpoint_patch():
    encoded = base64.b64encode(Path(patch.__file__).read_bytes()).decode()
    assert not any(
        encoded in command for command in Kimi_K3_LoRA_Recipe().image_run_commands
    )
    assert sum(encoded in command for command in build_recipe().image_run_commands) == 1
