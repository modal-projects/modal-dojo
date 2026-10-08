from __future__ import annotations

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_sglang_parallel_aliases as patcher,
)


def test_patch_adds_parallel_size_aliases_and_is_idempotent(tmp_path) -> None:
    target = tmp_path / "arguments.py"
    target.write_text(
        "def validate_args(args):\n"
        "    args.sglang_dp_size = args.sglang_data_parallel_size\n"
        "    args.sglang_pp_size = args.sglang_pipeline_parallel_size\n"
        "    args.sglang_ep_size = args.sglang_expert_parallel_size\n"
    )

    patcher._patch_file(target)
    patched = target.read_text()

    assert patcher.MARKER in patched
    assert "args.sglang_data_parallel_size = args.sglang_dp_size" in patched
    assert "args.sglang_pipeline_parallel_size = args.sglang_pp_size" in patched
    assert "args.sglang_expert_parallel_size = args.sglang_ep_size" in patched

    patcher._patch_file(target)
    assert target.read_text() == patched
