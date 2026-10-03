"""Pinned SGLang update contract; actual tensor copies are checked on H200."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_k3_lora_cpu_staging as patch,
)

SOURCE = Path(__file__).parent / "testdata/sglang/k3_lora_staging.py.input"


def test_patch_changes_only_staging_and_is_idempotent(tmp_path):
    target = tmp_path / "weight_updater.py"
    original = SOURCE.read_text()
    target.write_text(original)
    patch.apply(target)
    first = target.read_text()
    assert first.replace(patch.REPLACEMENT, patch.ANCHOR) == original
    patch.apply(target)
    assert target.read_text() == first


@pytest.mark.parametrize("change", ["missing", "duplicate", "partial"])
def test_unexpected_source_fails_without_mutation(tmp_path, change):
    source = SOURCE.read_text()
    if change == "missing":
        source = source.replace(patch.ANCHOR, "")
    elif change == "duplicate":
        source += source
    else:
        source += "\n# " + patch.MARKER
    target = tmp_path / "weight_updater.py"
    target.write_text(source)
    with pytest.raises(RuntimeError, match="unexpected"):
        patch.apply(target)
    assert target.read_text() == source


@pytest.fixture
def updater(tmp_path):
    target = tmp_path / "weight_updater.py"
    target.write_text(SOURCE.read_text())
    patch.apply(target)
    namespace = {
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(barrier=lambda **kwargs: None)
        ),
        "EndWeightUpdateReqOutput": SimpleNamespace,
        # Values stand in for digests: copy/byte integrity is exercised with
        # real tensors by the pinned-runtime H200 probe.
        "_sha256_tensor": lambda tensor: tensor,
    }
    exec("from __future__ import annotations\n" + target.read_text(), namespace)
    state = namespace["SchedulerWeightUpdaterManager"]()
    state._lora_stash = {"policy": {"A": "digest-a", "B": "digest-b"}}
    state._lora_applied_names = {}
    state._weight_update_in_progress = True
    state._weight_update_sync_base = False
    state._weight_update_pending_version = "2"
    state.tp_cpu_group = None
    installed, versions = [], []

    def install(name, tensors):
        installed.append((name, tensors.copy()))
        return SimpleNamespace(success=True)

    state.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            lora_manager=SimpleNamespace(apply_streamed_adapter=install)
        )
    )
    state.record_weight_version_after_update = versions.append
    return state, installed, versions


@pytest.mark.parametrize("case", ["valid", "abort", "checksum", "partial", "manifest"])
def test_end_update_preserves_validation_and_abort_contract(updater, case):
    state, installed, versions = updater
    checksums = {"policy": {"A": "digest-a", "B": "digest-b"}}
    if case == "checksum":
        checksums["policy"]["A"] = "wrong"
    elif case == "partial":
        state._lora_applied_names = {"policy": frozenset({"A", "B", "C"})}
    elif case == "manifest":
        checksums["unexpected"] = {}
    result = state.end_weight_update(
        SimpleNamespace(abort=case == "abort", expected_lora_checksums=checksums)
    )
    assert result.success == (case in {"valid", "abort"})
    assert not state._weight_update_in_progress
    assert state._weight_update_pending_version is None
    if case == "valid":
        assert installed == [("policy", {"A": "digest-a", "B": "digest-b"})]
        assert versions == ["2"]
        assert state._lora_stash == {}
    else:
        assert installed == versions == []
    if case == "abort":
        assert state._lora_stash == {}
