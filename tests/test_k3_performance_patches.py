from pathlib import Path

import pytest

from modal_dojo.frameworks.miles.modal_helpers.patches import (
    patch_sglang_offload_timing as timing,
    patch_tms_retain_backup as backup,
)

FIXTURES = Path(__file__).parent / "testdata" / "k3"


@pytest.mark.parametrize(
    "patcher, filename", [(timing, "weight_updater.py"), (backup, "tms_core.cpp")]
)
def test_pinned_source_and_idempotence(tmp_path, patcher, filename):
    target = tmp_path / filename
    source = (FIXTURES / (filename + ".input")).read_text()
    target.write_text(source)
    patcher.apply(target)
    patched = target.read_text()
    assert patched != source
    patcher.apply(target)
    assert target.read_text() == patched
    if patcher is timing:
        compile(patched, filename, "exec")
        # Instrumentation must preserve every operation, including the number
        # of barriers/synchronizations: extra ones would distort the profile.
        for operation in (
            ".synchronize()",
            "barrier(self.tp_cpu_group)",
            ".pause(",
            ".resume(",
            "_export_static_state(",
            "_import_static_state(",
        ):
            assert patched.count(operation) == source.count(operation)
    else:
        # Retain the allocation only: both transfers must remain unconditional
        # so updated weights and mutable model buffers are never restored stale.
        assert patched.count("CUDA_ERROR_CHECK(cudaMemcpy(") == source.count(
            "CUDA_ERROR_CHECK(cudaMemcpy("
        )
        assert "metadata.tag != retained_tag" in patched
        assert "std::getenv" in patched


def test_native_patch_rejects_drift_without_writing(tmp_path):
    target = tmp_path / "core.cpp"
    target.write_text("unknown version")
    with pytest.raises(RuntimeError, match="unexpected"):
        backup.apply(target)
    assert target.read_text() == "unknown version"
