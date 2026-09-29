from __future__ import annotations

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_qwen3_5_hf_dispatch as patcher,
)


def test_qwen36_hf_ids_match_qwen35_family() -> None:
    assert patcher.is_qwen35_family("Qwen/Qwen3.6-35B-A3B")
    assert patcher.is_qwen35_family("qwen3.5-35B-A3B")
    assert patcher.is_qwen35_family("Qwen/Qwen3.6-27B")
    assert patcher.is_qwen35_family("qwen3_5")
    assert not patcher.is_qwen35_family("Qwen/Qwen3-8B")
    assert not patcher.is_qwen35_family("zai-org/GLM-4.7")


def test_patch_routes_qwen36_and_is_idempotent(tmp_path) -> None:
    target = tmp_path / "__init__.py"
    target.write_text(
        "def _convert_to_hf_core(args, model_name, name, param):\n"
        '    if "qwen35" in model_name:\n'
        "        return convert_qwen3_5_to_hf(args, name, param)\n"
        '    elif "qwen3" in model_name:\n'
        "        return convert_qwen2_to_hf(args, name, param)\n"
    )

    patcher._patch_file(target)
    patched = target.read_text()
    assert patcher.MARKER in patched
    assert patcher.ALIGN_MARKER in patched
    assert "qwen36" in patched
    assert patched.index("qwen36") < patched.index('elif "qwen3" in model_name:')
    assert "WEIGHT_SYNC_IN" in patched

    patcher._patch_file(target)
    assert target.read_text() == patched
