"""Cluster-scheduling helpers in common/ray_cluster: the RDMA-by-GPU-family rule,
the single-node identity path of clustered_if, and PYTHONPATH handoff into Ray jobs.
"""

import pytest

from modal_dojo.common.ray_cluster import (
    _supports_rdma,
    _with_container_pythonpath,
    clustered_if,
)


@pytest.mark.parametrize(
    "gpu", ["H100", "H200", "B200", "B300", "GB200", "H100:8", "h100"]
)
def test_supports_rdma_true(gpu):
    assert _supports_rdma(gpu)


@pytest.mark.parametrize("gpu", ["A100", "L40S", "A10G", "T4", "", "A100:8"])
def test_supports_rdma_false(gpu):
    assert not _supports_rdma(gpu)


def test_clustered_if_single_node_is_identity():
    def fn():
        return None

    # Single node: no @clustered, just a plain registration — fn returned unchanged.
    assert clustered_if(False, 1, gpu_type="H100")(fn) is fn


def test_recipe_pythonpath_keeps_container_entries(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/pkg/:/root/")
    runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "MASTER_ADDR": "x"}}

    merged = _with_container_pythonpath(runtime_env)

    assert merged["env_vars"]["PYTHONPATH"] == "/root/Megatron-LM/:/pkg/:/root/"
    assert merged["env_vars"]["MASTER_ADDR"] == "x"
    assert runtime_env["env_vars"]["PYTHONPATH"] == "/root/Megatron-LM/"


def test_container_pythonpath_added_when_job_sets_none(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/pkg/:/root/")

    merged = _with_container_pythonpath({"env_vars": {"MASTER_ADDR": "x"}})

    assert merged["env_vars"]["PYTHONPATH"] == "/pkg/:/root/"


def test_duplicate_pythonpath_entries_collapse(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/pkg/:/root/")

    merged = _with_container_pythonpath({"env_vars": {"PYTHONPATH": "/root/:/extra"}})

    assert merged["env_vars"]["PYTHONPATH"] == "/root/:/extra:/pkg/"


def test_no_container_pythonpath_is_a_noop(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    runtime_env = {"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}

    assert _with_container_pythonpath(runtime_env) == runtime_env
