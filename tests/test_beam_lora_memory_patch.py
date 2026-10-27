from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches.patch_beam_lora_memory import (
    patch_source,
)

FIXTURES = Path(__file__).parent / "testdata/miles/beam"


def test_grouped_adapter_memory_patch_snapshot():
    original = (FIXTURES / "grouped_adapter.forward.input").read_text()
    patched = patch_source(original)
    assert patched == (FIXTURES / "grouped_adapter.forward.output").read_text()
    assert patch_source(patched) == patched


def test_grouped_adapter_patch_rejects_unknown_source():
    with pytest.raises(RuntimeError):
        patch_source("changed upstream")
