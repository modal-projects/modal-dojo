from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_fused_activation as patch,
)

SOURCE = Path(__file__).parent / "testdata/miles/k3_ops.py.input"


def test_fuses_original_formula_and_preserves_surrounding_source(tmp_path):
    target = tmp_path / "ops.py"
    original = SOURCE.read_text()
    target.write_text(original)
    patch.apply(target)
    first = target.read_text()
    assert first == SOURCE.with_suffix(".output").read_text()
    assert first.replace(patch.REPLACEMENT, patch.ANCHOR) == original
    patch.apply(target)
    assert target.read_text() == first


@pytest.mark.parametrize(
    "source",
    [
        "",
        patch.ANCHOR * 2,
        patch.MARKER + patch.ANCHOR,
        patch.REPLACEMENT + patch.ANCHOR,
    ],
)
def test_unexpected_source_is_not_modified(tmp_path, source):
    target = tmp_path / "ops.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source
