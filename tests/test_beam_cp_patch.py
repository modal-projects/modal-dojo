import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches.patch_beam_cp import patch_source

FIXTURES = Path(__file__).parent / "testdata/miles/beam"


def test_pinned_bridge_cp_patch_snapshot():
    original = (FIXTURES / "model_provider.py.input").read_text()
    patched = patch_source(original)
    assert patched == (FIXTURES / "model_provider.py.cp.output").read_text()
    assert patch_source(patched) == patched
    tree = ast.parse(patched)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_bridge_runtime_config"
    )
    # Exercise the real patched parallelism statements without importing GPU libraries.
    body = fn.body[1:8]
    for value in (["all_gather"], ["p2p"], None):
        provider = SimpleNamespace(cp_comm_type="existing")
        args = SimpleNamespace(
            **{
                k: 2
                for k in (
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "expert_model_parallel_size",
                    "expert_tensor_parallel_size",
                    "sequence_parallel",
                    "context_parallel_size",
                )
            },
            cp_comm_type=value,
        )
        exec(
            compile(ast.Module(body=body, type_ignores=[]), "<patched>", "exec"),
            {"provider": provider, "args": args},
        )
        assert provider.cp_comm_type == (value if value is not None else "existing")


def test_patch_rejects_unknown_source():
    with pytest.raises(RuntimeError):
        patch_source("unsupported upstream source")
