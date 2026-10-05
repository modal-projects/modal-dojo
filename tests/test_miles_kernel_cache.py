import json
from pathlib import Path
import shutil

from modal_dojo.frameworks.miles.kernel_cache import KernelCache, prepare_kernel_cache


def test_disabled_is_noop():
    original = {"TRITON_CACHE_DIR": "/tmp/configured"}
    mapped, cache = prepare_kernel_cache(original, node_rank=0, gpu_type="B300")
    assert mapped == original
    assert cache is None


def test_legacy_migration_and_snapshot_restore(tmp_path):
    persistent = tmp_path / "persistent"
    triton = persistent / "triton"
    key = triton / "hash"
    key.mkdir(parents=True)
    binary = key / "kernel.cubin"
    binary.write_bytes(b"compiled")
    (key / "__grp__kernel.json").write_text(
        json.dumps({"child_paths": {"cubin": str(binary)}})
    )
    (key / "tmp.pid_123").mkdir()
    (key / "tmp.pid_123" / "partial").write_text("partial")
    env = {
        "TRITON_CACHE_DIR": str(triton),
        "SGLANG_CACHE_DIR": str(persistent / "sglang"),
        "DOJO_LOCAL_KERNEL_CACHE": str(tmp_path / "local"),
    }
    cache = KernelCache(env, node_rank=0, gpu_type="B300")
    mapped = cache.prepare()
    local = Path(mapped["TRITON_CACHE_DIR"])
    manifest = json.loads((local / "hash" / "__grp__kernel.json").read_text())
    assert manifest["child_paths"]["cubin"] == str(local / "hash" / "kernel.cubin")
    assert not (local / "hash" / "tmp.pid_123").exists()
    (local / "new.autotune.json").write_text('{"configs_timings": []}')
    cache.snapshot()
    assert cache.archive.is_file()
    shutil.rmtree(cache.root)
    # A future container restores the new tuning result as well as binaries.
    next_cache = KernelCache(env, node_rank=0, gpu_type="B300")
    assert next_cache.prepare() == mapped
    assert (local / "new.autotune.json").is_file()
    assert Path(manifest["child_paths"]["cubin"]).read_bytes() == b"compiled"


def test_bad_archive_falls_back_to_legacy(tmp_path):
    env = {
        "TRITON_CACHE_DIR": str(tmp_path / "persistent" / "triton"),
        "DOJO_LOCAL_KERNEL_CACHE": str(tmp_path / "local"),
    }
    cache = KernelCache(env, node_rank=1, gpu_type="B300")
    cache.archive.parent.mkdir(parents=True)
    cache.archive.write_bytes(b"incomplete")
    assert cache.prepare()["TRITON_CACHE_DIR"].startswith(str(tmp_path / "local"))
