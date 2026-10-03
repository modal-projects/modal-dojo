"""Pinned allocation-source contract; numerical kernels are checked on H200."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_marlin_padding as patch,
)

SOURCE = Path(__file__).parent / "testdata/sglang/k3_mxfp4_create_weights.py.input"


def test_pinned_allocator_patch_is_idempotent_and_shape_scoped(tmp_path):
    target = tmp_path / "mxfp4.py"
    target.write_text(SOURCE.read_text())
    patch.apply(target)
    first = target.read_text()
    patch.apply(target)
    assert target.read_text() == first
    tree = ast.parse(first)
    method = tree.body[0].body[0]
    branch = next(n for n in method.body if isinstance(n, ast.If))
    # Execute the actual patched Marlin allocation branch, before tensor creation.
    code = compile(ast.Module(body=branch.body, type_ignores=[]), str(target), "exec")
    for hidden, intermediate, expected in [
        (3584, 192, 192),
        (3584, 384, 384),
        (4096, 192, 256),
        (3584, 96, 128),
    ]:
        method_state = SimpleNamespace()
        state = dict(
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            self=method_state,
            layer=SimpleNamespace(
                hidden_size=hidden, intermediate_size_per_partition=intermediate
            ),
            round_up=lambda x, multiple: (x + multiple - 1) // multiple * multiple,
        )
        exec(code, state)
        assert state["intermediate_size_per_partition_after_pad"] == expected
        assert method_state.intermediate_pad == expected - intermediate


@pytest.mark.parametrize("change", ["missing", "duplicate", "partial"])
def test_unexpected_source_fails_without_mutation(tmp_path, change):
    source = SOURCE.read_text()
    if change == "missing":
        source = source.replace(patch.ANCHOR, "")
    elif change == "duplicate":
        source += source
    else:
        source += "\n# " + patch.MARKER
    target = tmp_path / "mxfp4.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source
