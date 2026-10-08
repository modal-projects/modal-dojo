from __future__ import annotations

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_multimodal_prompt_filter as patcher,
)


def test_patch_reuses_parsed_multimodal_inputs_and_is_idempotent(tmp_path) -> None:
    data = tmp_path / "data.py"
    data.write_text(patcher.ANCHOR)

    patcher._patch_file(data)
    patched = data.read_text()

    assert patcher.MARKER in patched
    assert "multimodal_inputs = sample.multimodal_inputs" in patched
    assert "process_vision_info(sample.prompt" not in patched

    patcher._patch_file(data)
    assert data.read_text() == patched
