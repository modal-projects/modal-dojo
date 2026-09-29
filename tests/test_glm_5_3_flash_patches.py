"""Golden snapshots from the recipe's pinned Miles merge commit cc76e239."""

from pathlib import Path

import pytest

from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_glm_5_3_flash_fp8_device as fp8,
    patch_glm_5_3_flash_kda as kda,
    patch_glm_5_3_flash_timing as glm,
    patch_rollout_status_reporting as status,
)

SNAPSHOTS = Path(__file__).parent / "testdata/miles/glm_5_3_flash"


def test_fp8_reader_uses_rank_device_for_weights_and_separate_scale_shards():
    source = (SNAPSHOTS / "dequant_fp8_safetensor_io.py.input").read_text()
    patched = fp8.patch_source(source)
    assert patched == (SNAPSHOTS / "dequant_fp8_safetensor_io.py.output").read_text()
    compile(patched, "dequant_fp8_safetensor_io.py", "exec")
    assert fp8.patch_source(patched) == patched
    assert patched.count('device=f"cuda:{torch.cuda.current_device()}"') == 2
    with pytest.raises(ValueError, match="device anchors"):
        fp8.patch_source(source.replace('device="cuda"', 'device="cpu"', 1))


def test_kda_kernel_hoists_python_call_out_of_triton_jit():
    source = (SNAPSHOTS / "chunk_intra_token_parallel.py.input").read_text()
    patched = kda.patch_source(source)
    assert patched == (SNAPSHOTS / "chunk_intra_token_parallel.py.output").read_text()
    compile(patched, "chunk_intra_token_parallel.py", "exec")
    assert kda.patch_source(patched) == patched
    assert "BK: tl.constexpr = triton.next_power_of_2(K)" not in patched
    assert "BK=triton.next_power_of_2(K)" in patched


@pytest.mark.parametrize("name", ["train.py", "actor.py", "model.py"])
def test_pinned_timing_patch(name):
    source = (SNAPSHOTS / f"{name}.input").read_text()
    actual = glm.patch_source(source, name)
    assert actual == (SNAPSHOTS / f"{name}.output").read_text()
    compile(actual, name, "exec")
    assert glm.patch_source(actual, name) == actual


def test_driver_adapter_composes_with_shared_status_patch(tmp_path):
    target = tmp_path / "train.py"
    target.write_text((SNAPSHOTS / "train.py.input").read_text())
    status._patch_file(target)
    patched = glm.patch_source(target.read_text(), "train.py")
    compile(patched, "train.py", "exec")
    assert "_tg_glm_lane('driver', rollout_id)" in patched
    assert "_tg_glm_report('generate_rollouts', args, rollout_id)" in patched
    assert "_tg_glm_phase('initial_weight_sync')" in patched
    assert "_tg_glm_phase('weight_sync')" in patched


def test_changed_upstream_target_fails_instead_of_silently_losing_timings():
    source = (SNAPSHOTS / "model.py.input").read_text()
    source = source.replace("optimizer.step()", "optimizer.new_step()")
    with pytest.raises(ValueError, match="optimizer.step"):
        glm.patch_source(source, "model.py")
