from __future__ import annotations

import pytest

from modal_dojo.frameworks.slime.modal_helpers.patches import (
    patch_vlm_critic_value_head as patcher,
)

# Excerpts of the pinned image's slime sources (identical across every slime commit
# the image's source snapshots match).
MODEL_PROVIDER = """\
    if args.megatron_to_hf_mode == "bridge":
        provider.finalize()

        if role == "critic":
            _original_provide = provider.provide

            def _critic_provide(pre_process=True, post_process=True, vp_stage=None):
                model = _original_provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
                if post_process:
                    model.output_layer = LinearForLastLayer(
                        input_size=model.config.hidden_size, output_size=1, config=model.config
                    )
                return model

            return _critic_provide

        return provider.provide
"""
CHECKPOINT = """\
    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)
"""


def test_value_head_attaches_to_language_model_and_is_idempotent() -> None:
    patched = patcher.patch_model_provider(MODEL_PROVIDER)

    assert patcher.MARKER in patched
    assert 'head_owner = getattr(model, "language_model", None)' in patched
    assert "head_owner.output_layer = LinearForLastLayer(" in patched
    assert "model.output_layer = " not in patched
    compile("def _f():\n" + patched, "model_provider.py", "exec")
    assert patcher.patch_model_provider(patched) == patched


def test_critic_hf_load_skips_value_head_only_and_is_idempotent() -> None:
    patched = patcher.patch_checkpoint(CHECKPOINT)

    assert patcher.MARKER in patched
    assert 'getattr(ddp_model[0], "role", None) == "critic"' in patched
    assert (
        'AutoMapping.register_module_type("LinearForLastLayer", "replicated")'
        in patched
    )
    assert (
        'allowed_mismatched_params=["*output_layer.weight"] if is_critic else None'
        in patched
    )
    compile("def _f():\n" + patched, "checkpoint.py", "exec")
    assert patcher.patch_checkpoint(patched) == patched


def test_sources_without_bridge_path_are_left_alone() -> None:
    no_bridge = "def _get_model_provider_func(args, role):\n    return model_provider\n"
    no_hf_load = "def _load_checkpoint_megatron(ddp_model):\n    pass\n"

    assert patcher.patch_model_provider(no_bridge) == no_bridge
    assert patcher.patch_checkpoint(no_hf_load) == no_hf_load


def test_changed_bridge_code_fails_loudly() -> None:
    drifted_provider = MODEL_PROVIDER.replace(
        "output_size=1", "output_size=1, bias=False"
    )
    drifted_checkpoint = CHECKPOINT.replace(
        "(ddp_model)\n", "(ddp_model, strict=True)\n"
    )

    with pytest.raises(RuntimeError):
        patcher.patch_model_provider(drifted_provider)
    with pytest.raises(RuntimeError):
        patcher.patch_checkpoint(drifted_checkpoint)
