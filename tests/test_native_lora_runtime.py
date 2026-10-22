"""Pinned-source patch contract and native adapter continuation semantics."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles import native_lora_state as state
from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_native_lora_diagnostics,
    patch_native_lora_training,
    patch_native_lora_training_state,
)

FIXTURES = Path(__file__).parent / "testdata/miles_native_lora"


def test_pinned_patch_pipeline_is_complete_and_idempotent(tmp_path, request):
    snapshot = json.loads((FIXTURES / "SOURCE.json").read_text())
    for name, digest in snapshot["files"].items():
        source = (FIXTURES / (name + ".input")).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    patches = (
        patch_native_lora_training,
        patch_native_lora_training_state,
        patch_native_lora_diagnostics,
    )
    for patch in patches:
        patch.apply(tmp_path)
    outputs = {}
    for name in snapshot["files"]:
        result = (tmp_path / name).read_text()
        compile(result, name, "exec")
        assert "from tau_quest" not in result and "import tau_quest" not in result
        outputs[name] = result
        golden = FIXTURES / (name + ".output")
        if request.config.getoption("--rewrite"):
            golden.write_text(result)
        assert result == golden.read_text()
    for patch in patches:
        patch.apply(tmp_path)
    assert outputs == {name: (tmp_path / name).read_text() for name in outputs}


def test_patch_refuses_drift_in_the_loss_normalizer(tmp_path):
    target = tmp_path / "backends/training_utils/loss/objective.py"
    target.parent.mkdir(parents=True)
    target.write_text("# unexpected upstream implementation\n")
    with pytest.raises(ValueError, match="anchor changed"):
        patch_native_lora_training.patch_token_normalizer(tmp_path)


@pytest.mark.parametrize("fresh_optimizer", [False, True])
def test_base_loading_preserves_explicit_adapter_resume_intent(fresh_optimizer):
    args = SimpleNamespace(
        lora_adapter_path="/checkpoint/adapter",
        no_load_optim=fresh_optimizer,
        no_load_rng=False,
    )
    state.preserve_resume_flags(args)
    args.no_load_optim = args.no_load_rng = True
    state.prepare_native_resume(args)
    assert args.no_load_optim == fresh_optimizer and not args.no_load_rng
    with pytest.raises(ValueError, match="did not preserve"):
        state.prepare_native_resume(SimpleNamespace())


def test_native_resume_resolves_parent_sampler_and_checks_index():
    args = SimpleNamespace(
        load="/frozen-base", lora_adapter_path="/run/iter_0000005/adapter"
    )
    assert state.resume_run_directory(args, 5) == "/run"
    with pytest.raises(ValueError, match="index disagree"):
        state.resume_run_directory(args, 6)
    args.lora_adapter_path = None
    assert state.resume_run_directory(args, 5) == "/frozen-base"


def test_rng_restore_rejects_missing_state_and_changed_topology(monkeypatch):
    monkeypatch.setattr(state, "_rank_world", lambda: (0, 8))
    monkeypatch.setattr(state, "_layout", lambda: {"tensor_model_parallel_size": 4})
    with pytest.raises(ValueError, match="saved RNG"):
        state.restore_resume_state(None)
    with pytest.raises(ValueError, match="rank/topology"):
        state.restore_resume_state({"rank_world": [0, 4], "layout": {}})


def test_image_patches_have_no_harness_dependency_and_optional_reuse_is_off():
    import base64

    from modal_dojo import DeepSeek_V4_1_Flash_LoRA_Recipe

    recipe = DeepSeek_V4_1_Flash_LoRA_Recipe()
    sources = [
        base64.b64decode(command.split()[1]).decode()
        for command in recipe.image_run_commands
    ]
    assert any("patch_token_normalizer" in source for source in sources)
    assert any("capture_resume_state" in source for source in sources)
    assert any("lora_diagnostics" in source for source in sources)
    assert all(
        "import tau_quest" not in source and "from tau_quest" not in source
        for source in sources
    )
    assert not any("TMS_ARCHIVE_SHA256" in source for source in sources)
