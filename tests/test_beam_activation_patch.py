import ast

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches.patch_beam_activation import (
    ANCHOR,
    HELPER,
    patch_source,
)


def test_activation_patch_is_parseable_idempotent_and_fail_closed():
    source = (
        "class TEGroupedMLP(MegatronModule):\n    def forward(self):\n"
        + ANCHOR
        + "            pass\n"
    )
    patched = patch_source(source)
    ast.parse(patched)
    assert HELPER in patched
    assert patch_source(patched) == patched
    with pytest.raises(RuntimeError):
        patch_source("changed upstream")
