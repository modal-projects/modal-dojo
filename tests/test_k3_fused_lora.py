from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_fused_lora as patch,
)

SOURCE = Path(__file__).parent / "testdata/miles/k3_lora.py.input"


def test_fuses_expert_add_and_preserves_surrounding_source(tmp_path):
    target = tmp_path / "lora.py"
    original = SOURCE.read_text()
    target.write_text(original)
    patch.apply(target)
    first = target.read_text()
    assert first == SOURCE.with_suffix(".output").read_text()
    assert (
        first.replace(patch.HELPER, "").replace(patch.REPLACEMENT, patch.ANCHOR)
        == original
    )
    patch.apply(target)
    assert target.read_text() == first


@pytest.mark.parametrize(
    "source",
    [
        "",
        patch.ANCHOR * 2,
        patch.MARKER + patch.ANCHOR,
        patch.HELPER + patch.HELPER_ANCHOR + patch.REPLACEMENT + patch.ANCHOR,
    ],
)
def test_unexpected_source_is_not_modified(tmp_path, source):
    target = tmp_path / "lora.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source
