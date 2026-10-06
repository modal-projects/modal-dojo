from __future__ import annotations

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_vlm_critic_value_head as patcher,
)


def test_patch_moves_value_head_and_relaxes_critic_load(tmp_path) -> None:
    provider = tmp_path / "model_provider.py"
    checkpoint = tmp_path / "checkpoint.py"
    provider.write_text(patcher.PROVIDER_ANCHOR)
    checkpoint.write_text(patcher.LOAD_ANCHOR)

    patcher._patch_file(provider, patcher.PROVIDER_ANCHOR, patcher.PROVIDER_REPLACEMENT)
    patcher._patch_file(checkpoint, patcher.LOAD_ANCHOR, patcher.LOAD_REPLACEMENT)
    patched_provider, patched_checkpoint = provider.read_text(), checkpoint.read_text()

    assert "head_owner.output_layer = LinearForLastLayer(" in patched_provider
    assert (
        'register_module_type("LinearForLastLayer", "replicated")' in patched_checkpoint
    )
    assert 'allowed_mismatched_params = ["*output_layer.weight"]' in patched_checkpoint

    patcher._patch_file(provider, patcher.PROVIDER_ANCHOR, patcher.PROVIDER_REPLACEMENT)
    assert provider.read_text() == patched_provider
