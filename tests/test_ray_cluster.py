"""Cluster-scheduling helpers in common/ray_cluster: the RDMA-by-GPU-family rule
and the single-node identity path of clustered_if.
"""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from modal_training_gym.common import ray_cluster
from modal_training_gym.common.ray_cluster import (
    ModalRayCluster,
    _supports_rdma,
    capture_ray_cluster_diagnostics,
    clustered_if,
    start_ray_worker,
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


def test_modal_cluster_identity_snapshot_is_rank_ordered_and_detached(monkeypatch):
    import modal.experimental

    monkeypatch.setattr(
        modal.experimental,
        "get_cluster_info",
        lambda: SimpleNamespace(
            rank=1,
            cluster_id="cluster-abc",
            container_ipv4_ips=["10.0.0.1", "10.0.0.2"],
        ),
    )

    cluster = ModalRayCluster()
    cluster.discover_cluster(2)
    identity = cluster.identity_snapshot()

    assert cluster.rank == 1
    assert cluster.node_ip == "10.0.0.2"
    assert identity == {
        "schema_version": 1,
        "cluster_id": "cluster-abc",
        "node_count": 2,
        "rank_ordered_container_ipv4_ips": ["10.0.0.1", "10.0.0.2"],
        "head_addr": "10.0.0.1",
    }
    identity["rank_ordered_container_ipv4_ips"].append("mutated")
    assert cluster.identity_snapshot()["rank_ordered_container_ipv4_ips"] == [
        "10.0.0.1",
        "10.0.0.2",
    ]


def test_single_node_identity_uses_safe_loopback_fallback(monkeypatch):
    import modal.experimental

    def _unexpected_cluster_lookup():
        raise AssertionError("size-1 discovery must not require Modal cluster info")

    monkeypatch.setattr(
        modal.experimental, "get_cluster_info", _unexpected_cluster_lookup
    )

    cluster = ModalRayCluster()
    cluster.discover_cluster(1)

    assert cluster.identity_snapshot() == {
        "schema_version": 1,
        "cluster_id": None,
        "node_count": 1,
        "rank_ordered_container_ipv4_ips": ["127.0.0.1"],
        "head_addr": "127.0.0.1",
    }


def test_cluster_identity_requires_discovery():
    with pytest.raises(RuntimeError, match="discover_cluster"):
        ModalRayCluster().identity_snapshot()


def test_cluster_member_identity_binds_rank_ip_and_modal_task(monkeypatch, capsys):
    import modal.experimental

    monkeypatch.setattr(
        modal.experimental,
        "get_cluster_info",
        lambda: SimpleNamespace(
            rank=1,
            cluster_id="cluster-abc",
            container_ipv4_ips=["10.0.0.1", "10.0.0.2"],
        ),
    )
    monkeypatch.setenv("MODAL_TASK_ID", "ta-worker-1")
    cluster = ModalRayCluster()
    cluster.discover_cluster(2)

    assert cluster.emit_member_identity(training_run_id="run-abc") == {
        "schema_version": 1,
        "training_run_id": "run-abc",
        "cluster_id": "cluster-abc",
        "node_count": 2,
        "rank": 1,
        "container_ipv4_ip": "10.0.0.2",
        "task_id": "ta-worker-1",
    }
    output = capsys.readouterr().out
    assert output.startswith("TRAINING_GYM_MODAL_CLUSTER_MEMBER ")
    assert '"task_id":"ta-worker-1"' in output


def test_ray_worker_waits_for_head_and_retries_failed_start(monkeypatch):
    connections = []
    starts = []
    stops = []
    sleeps = []

    class _Connection:
        def close(self):
            connections.append("closed")

    connect_results = [
        ConnectionRefusedError("head not ready"),
        _Connection(),
    ]

    def _connect(*_args, **_kwargs):
        result = connect_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    start_results = [
        SimpleNamespace(returncode=1, stdout="", stderr="GCS unavailable"),
        SimpleNamespace(returncode=0, stdout="started", stderr=""),
    ]

    monkeypatch.setattr(ray_cluster.socket, "create_connection", _connect)

    def _run(command, **_kwargs):
        if command[:2] == ["ray", "stop"]:
            stops.append(command)
            return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
        starts.append(command)
        return start_results.pop(0)

    monkeypatch.setattr(ray_cluster.subprocess, "run", _run)
    monkeypatch.setattr(
        ray_cluster.time,
        "sleep",
        sleeps.append,
    )

    start_ray_worker(
        "10.0.0.2",
        "10.0.0.1",
        connect_retries=3,
        retry_interval_seconds=0.25,
    )

    assert connections == ["closed"]
    assert len(starts) == 2
    assert stops == [["ray", "stop", "--force"]]
    assert starts[0][-2:] == ["--address", "10.0.0.1:6379"]
    assert sleeps == [0.25, 0.25]


def test_ray_worker_surfaces_bounded_last_start_failure(monkeypatch):
    class _Connection:
        def close(self):
            return None

    monkeypatch.setattr(
        ray_cluster.socket,
        "create_connection",
        lambda *_args, **_kwargs: _Connection(),
    )

    def _run(command, **_kwargs):
        if command[:2] == ["ray", "stop"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=17,
            stdout="",
            stderr="x" * 5_000,
        )

    monkeypatch.setattr(ray_cluster.subprocess, "run", _run)
    monkeypatch.setattr(ray_cluster.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="exited with code 17") as exc_info:
        start_ray_worker(
            "10.0.0.2",
            "10.0.0.1",
            connect_retries=1,
            start_retries=2,
            retry_interval_seconds=0,
        )

    assert len(str(exc_info.value)) < 4_200


def test_ray_worker_cleans_partial_state_after_start_timeout(monkeypatch):
    commands = []

    class _Connection:
        def close(self):
            return None

    monkeypatch.setattr(
        ray_cluster.socket,
        "create_connection",
        lambda *_args, **_kwargs: _Connection(),
    )

    starts = 0

    def _run(command, **kwargs):
        nonlocal starts
        commands.append((command, kwargs.get("timeout")))
        if command[:2] == ["ray", "stop"]:
            return SimpleNamespace(returncode=0, stdout="stopped", stderr="")
        starts += 1
        if starts == 1:
            raise ray_cluster.subprocess.TimeoutExpired(
                command,
                kwargs["timeout"],
            )
        return SimpleNamespace(returncode=0, stdout="started", stderr="")

    monkeypatch.setattr(ray_cluster.subprocess, "run", _run)
    monkeypatch.setattr(ray_cluster.time, "sleep", lambda _seconds: None)

    start_ray_worker(
        "10.0.0.2",
        "10.0.0.1",
        connect_retries=1,
        start_retries=2,
        start_timeout_seconds=60,
        retry_interval_seconds=0,
    )

    assert commands == [
        (
            [
                "ray",
                "start",
                "--node-ip-address=10.0.0.2",
                "--address",
                "10.0.0.1:6379",
            ],
            60,
        ),
        (["ray", "stop", "--force"], 30),
        (
            [
                "ray",
                "start",
                "--node-ip-address=10.0.0.2",
                "--address",
                "10.0.0.1:6379",
            ],
            60,
        ),
    ]


def test_stop_ray_is_bounded_and_marks_cluster_stopped(monkeypatch):
    calls = []
    shutdown = []
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(shutdown=lambda: shutdown.append(True))
    )

    def _run(command, **kwargs):
        assert shutdown == [True], "Disconnect the driver before killing its GCS server"
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")

    monkeypatch.setattr(ray_cluster.subprocess, "run", _run)
    cluster = ModalRayCluster()
    cluster._started = True

    cluster.stop_ray(timeout_seconds=7)

    assert calls == [
        (
            ["ray", "stop", "--force"],
            {
                "capture_output": True,
                "check": False,
                "text": True,
                "timeout": 7,
            },
        )
    ]
    assert cluster._started is False


def test_stop_ray_without_started_cluster_does_not_disconnect_other_driver(monkeypatch):
    def unexpected():
        raise AssertionError("An unused cluster must not disconnect another driver")

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(shutdown=unexpected))
    ModalRayCluster().stop_ray()


def test_sustained_failure_requires_one_continuous_interval():
    tracker = ray_cluster._SustainedFailure(grace_seconds=60)

    assert tracker.observe(failed=True, now=10) is False
    assert tracker.observe(failed=True, now=69) is False
    assert tracker.observe(failed=False, now=70) is False
    assert tracker.observe(failed=True, now=100) is False
    assert tracker.observe(failed=True, now=160) is True


def test_worker_tolerates_transient_head_loss_then_exits_after_grace(
    monkeypatch, capsys
):
    import modal.experimental

    monkeypatch.setattr(
        modal.experimental,
        "get_cluster_info",
        lambda: SimpleNamespace(
            rank=1,
            cluster_id="cluster-abc",
            container_ipv4_ips=["10.0.0.1", "10.0.0.2"],
        ),
    )
    checks = iter([OSError("transient"), object(), OSError("down"), OSError("down")])

    class _Connection:
        def close(self):
            return None

    def _connect(*_args, **_kwargs):
        result = next(checks)
        if isinstance(result, OSError):
            raise result
        return _Connection()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(ray_cluster.socket, "create_connection", _connect)
    monkeypatch.setattr(ray_cluster.asyncio, "sleep", _no_sleep)
    cluster = ModalRayCluster()
    cluster.discover_cluster(2)
    clock = iter((0.0, 0.25, 10.0, 11.0))
    monkeypatch.setattr(cluster, "_monotonic", lambda: next(clock))

    asyncio.run(
        cluster.wait_forever(
            poll_seconds=0.25,
            head_failure_grace_seconds=1.0,
        )
    )

    output = capsys.readouterr().out
    assert "Ray head liveness recovered" in output
    assert "remained unreachable through the grace period" in output


def test_worker_exits_cleanly_on_authenticated_post_success_head_shutdown(
    monkeypatch, capsys
):
    async def _no_sleep(_seconds):
        return None

    probes = []
    stops = []

    async def _accepted_completion_probe():
        probes.append("checked")
        return True

    monkeypatch.setattr(
        ray_cluster.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("head stopped")),
    )
    monkeypatch.setattr(ray_cluster.asyncio, "sleep", _no_sleep)
    cluster = ModalRayCluster()
    cluster.rank = 1
    cluster.head_addr = "10.0.0.1"
    cluster._discovered = True
    monkeypatch.setattr(cluster, "stop_ray", lambda: stops.append("stopped"))

    asyncio.run(
        cluster.wait_forever(
            poll_seconds=0.25,
            head_failure_grace_seconds=60,
            accepted_completion_probe=_accepted_completion_probe,
        )
    )

    assert probes == ["checked"]
    assert stops == ["stopped"]
    output = capsys.readouterr().out
    assert "Authenticated accepted TrainResult" in output
    assert "grace period" not in output


def test_worker_probe_error_preserves_sustained_head_failure_path(monkeypatch, capsys):
    async def _no_sleep(_seconds):
        return None

    async def _inconclusive_completion_probe():
        raise RuntimeError("acceptance record malformed")

    def _head_unreachable(*_args, **_kwargs):
        raise OSError("head stopped")

    monkeypatch.setattr(ray_cluster.socket, "create_connection", _head_unreachable)
    monkeypatch.setattr(ray_cluster.asyncio, "sleep", _no_sleep)
    cluster = ModalRayCluster()
    cluster.rank = 1
    cluster.head_addr = "10.0.0.1"
    cluster._discovered = True
    clock = iter((0.0, 1.0))
    monkeypatch.setattr(cluster, "_monotonic", lambda: next(clock))

    asyncio.run(
        cluster.wait_forever(
            poll_seconds=0.25,
            head_failure_grace_seconds=0.5,
            accepted_completion_probe=_inconclusive_completion_probe,
        )
    )

    output = capsys.readouterr().out
    assert "probe was inconclusive" in output
    assert "remained unreachable through the grace period" in output
    assert "exiting worker cleanly" not in output


def test_head_reports_only_sustained_worker_loss(monkeypatch):
    import modal.experimental

    monkeypatch.setattr(
        modal.experimental,
        "get_cluster_info",
        lambda: SimpleNamespace(
            rank=0,
            cluster_id="cluster-abc",
            container_ipv4_ips=["10.0.0.1", "10.0.0.2"],
        ),
    )
    snapshots = iter(
        (
            [{"Alive": True}, {"Alive": False}],
            [{"Alive": True}, {"Alive": True}],
            [{"Alive": True}, {"Alive": False}],
            [{"Alive": True}, {"Alive": False}],
        )
    )
    cluster = ModalRayCluster()
    cluster.discover_cluster(2)
    clock = iter((0.0, 1.0, 10.0, 70.0))
    monkeypatch.setattr(cluster, "_monotonic", lambda: next(clock))

    async def _nodes_snapshot():
        return next(snapshots)

    monkeypatch.setattr(cluster, "_ray_nodes_snapshot", _nodes_snapshot)

    message = asyncio.run(
        cluster._wait_for_sustained_worker_loss(
            poll_seconds=0.001,
            failure_grace_seconds=60.0,
        )
    )

    assert message == "Ray cluster lost worker nodes for at least 60s: 1/2 alive"


def test_failure_diagnostics_preserve_dead_node_reason(monkeypatch):
    fake_ray = SimpleNamespace(
        nodes=lambda: [
            {
                "NodeID": "head",
                "Alive": True,
                "NodeManagerAddress": "10.0.0.1",
                "Resources": {"GPU": 8.0},
            },
            {
                "NodeID": "worker",
                "Alive": False,
                "NodeManagerAddress": "10.0.0.2",
                "DeathReason": "UNEXPECTED_TERMINATION",
                "DeathReasonMessage": "raylet missed heartbeats",
            },
        ],
        cluster_resources=lambda: {"GPU": 16.0},
        available_resources=lambda: {"GPU": 8.0},
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        ray_cluster.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="two nodes, one dead",
            stderr="",
            returncode=0,
        ),
    )

    diagnostics = capture_ray_cluster_diagnostics()

    assert diagnostics["nodes"][1]["Alive"] is False
    assert diagnostics["nodes"][1]["DeathReasonMessage"] == "raylet missed heartbeats"
    assert diagnostics["cluster_resources"] == {"GPU": 16.0}
    assert diagnostics["ray_status"] == "two nodes, one dead"


def test_submitted_job_id_survives_log_stream_failure():
    class _Client:
        def submit_job(self, **_kwargs):
            return "ray-job-before-stream-failure"

        def tail_job_logs(self, _job_id):
            raise ConnectionError("dashboard disconnected")

    cluster = ModalRayCluster()
    cluster.discover_cluster(1)
    cluster._client = _Client()

    with pytest.raises(ConnectionError, match="dashboard disconnected"):
        asyncio.run(cluster.submit_and_tail("python train.py"))

    assert cluster.last_submitted_job_id == "ray-job-before-stream-failure"


def test_submit_aborts_and_records_diagnostics_after_sustained_worker_loss(
    monkeypatch,
):
    stopped_jobs = []

    class _Client:
        def submit_job(self, **_kwargs):
            return "ray-job-worker-loss"

        async def tail_job_logs(self, _job_id):
            await asyncio.Event().wait()

        def stop_job(self, job_id):
            stopped_jobs.append(job_id)

    async def _worker_loss():
        return "Ray cluster lost worker nodes for at least 60s: 1/3 alive"

    cluster = ModalRayCluster()
    cluster.n_nodes = 3
    cluster.rank = 0
    cluster._discovered = True
    cluster._client = _Client()
    monkeypatch.setattr(cluster, "_wait_for_sustained_worker_loss", _worker_loss)
    monkeypatch.setattr(
        ray_cluster,
        "capture_ray_cluster_diagnostics",
        lambda: {"nodes": [{"Alive": True}, {"Alive": False}]},
    )

    result = asyncio.run(cluster.submit_and_tail("python train.py"))

    assert result.status == "FAILED"
    assert result.is_success is False
    assert result.message == (
        "Ray cluster lost worker nodes for at least 60s: 1/3 alive"
    )
    assert result.diagnostics == {"nodes": [{"Alive": True}, {"Alive": False}]}
    assert stopped_jobs == ["ray-job-worker-loss"]


def test_failure_diagnostic_payload_is_bounded(monkeypatch):
    long_reason = "x" * 5_000
    fake_ray = SimpleNamespace(
        nodes=lambda: [
            {
                "NodeID": f"node-{index}",
                "Alive": False,
                "DeathReasonMessage": long_reason,
            }
            for index in range(300)
        ],
        cluster_resources=lambda: {
            f"resource-{index}": float(index) for index in range(300)
        },
        available_resources=lambda: {},
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        ray_cluster.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="s" * 13_000,
            stderr="",
            returncode=0,
        ),
    )

    diagnostics = capture_ray_cluster_diagnostics()

    assert len(diagnostics["nodes"]) == 256
    assert diagnostics["nodes_truncated"] == 44
    assert len(diagnostics["nodes"][0]["DeathReasonMessage"]) == 4_000
    assert len(diagnostics["cluster_resources"]) == 256
    assert len(diagnostics["ray_status"]) == 12_000


def test_failure_diagnostics_bound_wedged_ray_api(monkeypatch):
    release = threading.Event()

    def _blocked_nodes():
        release.wait(10)
        return []

    fake_ray = SimpleNamespace(
        nodes=_blocked_nodes,
        cluster_resources=lambda: {},
        available_resources=lambda: {},
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(
        ray_cluster,
        "_RAY_API_DIAGNOSTIC_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        ray_cluster.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="status still available",
            stderr="",
            returncode=0,
        ),
    )

    started = time.monotonic()
    diagnostics = capture_ray_cluster_diagnostics()
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 1
    assert "TimeoutError" in diagnostics["ray_api_error"]
    assert diagnostics["ray_status"] == "status still available"
