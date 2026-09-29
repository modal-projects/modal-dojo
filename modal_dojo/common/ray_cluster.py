"""Shared helpers for running a Ray cluster across Modal containers.

- `start_ray_head` / `start_ray_worker`: low-level primitives for spawning the
  Ray head/worker processes on a single container.
- `ModalRayCluster`: a base class that stitches those primitives together with
  Modal cluster discovery, the Ray `JobSubmissionClient`, dashboard port
  forwarding, and a worker keep-alive loop. Framework launchers (slime,
  verl, …) subclass or compose this.
"""

import asyncio
import inspect
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

from modal.experimental import clustered

from modal_dojo.train_recipes.gpu_allocation import _normalize_gpu_type

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265

_GCS_HEALTH_CHECK_ENV = {
    "RAY_health_check_period_ms": "10000",
    "RAY_health_check_timeout_ms": "20000",
    "RAY_health_check_failure_threshold": "60",
}

# GPU families with an RDMA/EFA fabric; other types don't support it and would fail
# if it were forced on.
_RDMA_GPU_TYPES = frozenset({"H100", "H200", "B200", "B300", "GB200"})


def _supports_rdma(gpu_type: str) -> bool:
    """Whether *gpu_type* (e.g. ``"H100"`` or ``"H100:8"``) is RDMA/EFA-capable."""
    return _normalize_gpu_type(gpu_type) in _RDMA_GPU_TYPES


def clustered_if(
    use_clustered: bool, size: int, *, gpu_type: str
) -> Callable[[Callable], Callable]:
    """Return ``clustered(size, rdma=…)`` when *use_clustered*, else an identity
    decorator that registers the function as a plain ``@app.function``.

    RDMA/EFA is enabled only when clustered (i.e. multi-node) *and* the GPU family
    supports it (H100/H200/B200/B300/GB200).
    """
    if use_clustered:
        return clustered(size, rdma=_supports_rdma(gpu_type))  # pyright: ignore[reportCallIssue, reportOptionalCall]

    def _identity(fn: Callable) -> Callable:
        return fn

    return _identity


def start_ray_head(
    node_ip: str,
    n_nodes: int,
    *,
    init_retries: int = 30,
    worker_wait_retries: int = 60,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start the Ray head and block until all workers have joined.

    Spawns `ray start --head` in the background, then polls `ray.init()` until
    the head is reachable and `ray.nodes()` until `n_nodes` are alive. Raises
    `RuntimeError` if either poll times out.
    """
    import ray

    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={node_ip}",
        "--dashboard-host=0.0.0.0",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)
    head_env = {**os.environ, **_GCS_HEALTH_CHECK_ENV}
    subprocess.Popen(cmd, env=head_env)

    for _ in range(init_retries):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Failed to connect to the Ray head node")

    for _ in range(worker_wait_retries):
        alive_nodes = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for Ray workers: {len(alive_nodes)}/{n_nodes} alive")
        if len(alive_nodes) == n_nodes:
            print("All Ray nodes connected")
            return
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def start_ray_worker(
    node_ip: str,
    head_addr: str,
    *,
    head_port: int = RAY_PORT,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start a Ray worker connected to the head at `head_addr:head_port`."""
    cmd = [
        "ray",
        "start",
        f"--node-ip-address={node_ip}",
        "--address",
        f"{head_addr}:{head_port}",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)
    subprocess.Popen(cmd)


def _with_container_pythonpath(runtime_env: dict) -> dict:
    container_pythonpath = os.environ.get("PYTHONPATH", "")
    if not container_pythonpath:
        return runtime_env

    env_vars = dict(runtime_env.get("env_vars") or {})
    entries = [
        entry for entry in env_vars.get("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    for entry in container_pythonpath.split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    env_vars["PYTHONPATH"] = os.pathsep.join(entries)
    return {**runtime_env, "env_vars": env_vars}


@dataclass
class ModalRayJobResult:
    status: str
    is_success: bool
    message: str | None = None


class ModalRayCluster:
    """Starts and coordinates a Ray cluster inside a Modal clustered function.

    Attributes:
        n_nodes: Total cluster node count.
        rank: This container's rank, where zero is the head.
        head_addr: IPv4 address of the head node.
        node_ip: IPv4 address of this container.
        is_head: Whether this container is the head.
    """

    n_nodes: int
    rank: int
    head_addr: str
    node_ip: str
    is_head: bool

    def __init__(self) -> None:
        # Populated by discover_cluster(); see _discovered for readiness check.
        self.n_nodes: int = 0
        self.rank: int = 0
        self.head_addr: str = ""
        self.node_ip: str = ""
        self._client = None
        self._discovered: bool = False
        self._started: bool = False

    def head_extra_start_args(self) -> list[str]:
        """Additional arguments for ``ray start --head``."""
        return []

    def worker_extra_start_args(self) -> list[str]:
        """Additional arguments for ``ray start`` on worker ranks."""
        return []

    @property
    def is_head(self) -> bool:
        if not self._discovered:
            raise RuntimeError(
                "ModalRayCluster.discover_cluster() has not been called yet"
            )
        return self.rank == 0

    @property
    def client(self):
        """Ray ``JobSubmissionClient`` for the head node, or ``None`` before startup.

        Raises:
            RuntimeError:
                Cluster discovery has not run or this container is not the head node.
        """
        if not self.is_head:
            raise RuntimeError("JobSubmissionClient is only available on the head node")
        return self._client

    def discover_cluster(self, n_nodes: int) -> None:
        """Discover this container's Ray cluster state.

        Args:
            n_nodes:
                Number of containers in the cluster.
        """
        if self._discovered:
            return

        if n_nodes == 1:
            # Modal may omit container IPv4s for size-1 clustered functions.
            rank, head_addr, node_ip = 0, "127.0.0.1", "127.0.0.1"
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            ips = list(info.container_ipv4_ips or [])
            if len(ips) != n_nodes:
                raise RuntimeError(
                    f"Modal cluster size mismatch: expected {n_nodes} nodes, got {len(ips)}"
                )
            rank = info.rank
            head_addr = ips[0]
            node_ip = ips[rank]

        self.n_nodes = n_nodes
        self.rank = rank
        self.head_addr = head_addr
        self.node_ip = node_ip
        self._discovered = True

    def start_ray(
        self,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Start the Ray head or worker.

        Args:
            init_retries:
                Maximum Ray head initialization attempts.
            worker_wait_retries:
                Maximum attempts to wait for workers.

        Raises:
            RuntimeError:
                ``discover_cluster()`` has not been called.
        """
        if self._started:
            return
        if not self._discovered:
            raise RuntimeError("discover_cluster() must be called before start_ray()")

        if self.rank == 0:
            start_ray_head(
                self.node_ip,
                self.n_nodes,
                init_retries=init_retries,
                worker_wait_retries=worker_wait_retries,
                extra_start_args=self.head_extra_start_args(),
            )
            from ray.job_submission import JobSubmissionClient

            self._client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
        else:
            start_ray_worker(
                self.node_ip,
                self.head_addr,
                extra_start_args=self.worker_extra_start_args(),
            )
        self._started = True

    def start(
        self,
        n_nodes: int,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Discover the cluster and start Ray."""
        self.discover_cluster(n_nodes)
        self.start_ray(
            init_retries=init_retries,
            worker_wait_retries=worker_wait_retries,
        )

    def forward_dashboard(self):
        """Open a forwarding context for the Ray dashboard.

        Returns:
            A context manager compatible with ``with`` and ``async with``.

        Raises:
            RuntimeError:
                This container is not the head node.
        """
        import modal

        if not self.is_head:
            raise RuntimeError("forward_dashboard is only valid on the head node")
        return modal.forward(RAY_DASHBOARD_PORT)

    async def submit_and_tail(
        self,
        entrypoint: str,
        *,
        runtime_env: dict | None = None,
        max_retries: int = 35,
    ) -> ModalRayJobResult:
        """Submit a Ray job and stream its logs.

        Returns:
            The final Ray job status.
        """
        if not self.is_head:
            raise RuntimeError("submit_and_tail is only valid on the head node")
        assert self._client is not None
        job_id = self._client.submit_job(
            entrypoint=entrypoint,
            runtime_env=_with_container_pythonpath(runtime_env or {}),
        )
        print(f"Submitted Ray job: {job_id}")

        _TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
        retry_count = 0

        async def _poll_status() -> str | None:
            try:
                job_status = self._client.get_job_status(job_id)
                if inspect.isawaitable(job_status):
                    job_status = await job_status
                return job_status.value
            except Exception as exc:  # noqa: BLE001
                print(f"\n[ray] Status poll failed; retrying: {exc}", flush=True)
                return None

        while retry_count < max_retries:

            async def _tail_logs() -> None:
                log_stream = self._client.tail_job_logs(job_id)
                if inspect.isawaitable(log_stream):
                    log_stream = await log_stream
                if hasattr(log_stream, "__aiter__"):
                    async for line in log_stream:
                        print(line, end="", flush=True)
                else:
                    for line in log_stream:
                        print(line, end="", flush=True)

            tail_task = asyncio.create_task(_tail_logs())
            try:
                while not tail_task.done():
                    status = await _poll_status()
                    if status in _TERMINAL:
                        done, _ = await asyncio.wait({tail_task}, timeout=5)
                        if done:
                            await tail_task
                        break
                    await asyncio.sleep(2)
                else:
                    await tail_task
                    status = await _poll_status() or "UNKNOWN"
            finally:
                if not tail_task.done():
                    tail_task.cancel()
                    try:
                        await tail_task
                    except asyncio.CancelledError:
                        pass

            if status in _TERMINAL:
                break
            retry_count += 1
            print(
                f"\n[ray] Log stream ended but job {job_id} is still {status}; "
                f"reconnecting in 2s... (retry {retry_count}/{max_retries})"
            )
            await asyncio.sleep(2)
        else:
            raise RuntimeError(
                f"Ray job {job_id} log stream disconnected {max_retries} times "
                f"without reaching terminal status (current: {status})"
            )

        print(f"\nFinal Ray job status: {status}")
        if status != "SUCCEEDED":
            # Surface Ray's recorded driver failure message in the exception
            # itself: the real traceback is streamed above but is easily buried
            # in rollout logs and is lost once logs roll off after termination.
            message = None
            try:
                info = self._client.get_job_info(job_id)
                if inspect.isawaitable(info):
                    info = await info
                message = getattr(info, "message", None) or None
            except Exception:  # noqa: BLE001 — best-effort enrichment
                pass
            suffix = f": {message}" if message else ""
            print(f"Ray job {job_id} finished with status: {status}{suffix}")
            return ModalRayJobResult(
                status=status, is_success=status == "SUCCEEDED", message=message
            )
        return ModalRayJobResult(status=status, is_success=status == "SUCCEEDED")

    async def wait_forever(self, poll_seconds: float = 10) -> None:
        """Keep a worker container alive until the head terminates the cluster."""
        while True:
            await asyncio.sleep(poll_seconds)
