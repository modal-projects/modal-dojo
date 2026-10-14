"""NCCL gang preflight run between cluster bring-up and framework start.

A cheap all_gather + all_to_all smoke test across every allocated GPU,
submitted as a short Ray job before the framework driver. A dead rank or
stalled link shows up here in ~seconds with the offending ranks named,
instead of as a 600s collective watchdog timeout deep into a training run.
"""

import shlex
from typing import TYPE_CHECKING

from .ray_cluster import ModalRayCluster

if TYPE_CHECKING:
    from ..train_recipes.gpu_allocation import GpuAllocation

_SMOKE_SRC = """\
import os
import sys
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

world = int(os.environ["PREFLIGHT_WORLD"])
timeout_s = float(os.environ["PREFLIGHT_TIMEOUT_S"])

ray.init(address="auto")


@ray.remote(num_gpus=1)
class _Probe:
    def run(self, rank):
        os.environ["MASTER_ADDR"] = os.environ["PREFLIGHT_MASTER_ADDR"]
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=timeout_s),
        )
        t = torch.ones(world, device="cuda")
        dist.all_gather([torch.empty_like(t) for _ in range(world)], t)
        dist.all_to_all_single(t, t)
        dist.barrier()
        return rank


probes = [_Probe.remote() for _ in range(world)]
refs = {
    probe.run.remote(rank): rank
    for rank, probe in enumerate(probes)
}
done, pending = ray.wait(
    list(refs), num_returns=len(refs), timeout=timeout_s
)

failures = {}
for ref in done:
    try:
        ray.get(ref)
    except Exception as e:
        failures[refs[ref]] = e

stalled = sorted(refs[ref] for ref in pending)
if failures or stalled:
    for rank, err in sorted(failures.items()):
        print(f"preflight rank {rank} failed: {err}")
    if stalled:
        print(f"preflight ranks stalled: {stalled}")
    sys.exit(1)

print(f"NCCL preflight OK: all_gather + all_to_all across {world} ranks")
"""


async def run_cluster_preflight(
    cluster: ModalRayCluster,
    allocation: "GpuAllocation",
    *,
    env: dict[str, str] | None = None,
    timeout_s: float = 120.0,
) -> None:
    """Smoke-test NCCL collectives across the gang before training starts.

    Skips single-GPU clusters. Raises ``RuntimeError`` naming any stalled or
    failed ranks when the smoke test does not succeed.

    Args:
        cluster: Started gang cluster (head node only).
        allocation: Resolved ``GpuAllocation``; every allocated GPU gets a
            probe so disaggregated rollout/critic GPUs are covered too.
        env: Extra environment (e.g. the recipe's training env vars) applied
            to the probes so the smoke test exercises the same NCCL
            configuration as training.
        timeout_s: Budget for probe init, collectives, and the driver wait.
    """
    world = allocation.total_gpus
    if world <= 1:
        return

    print(f"Preflighting NCCL across {world} ranks ({timeout_s:.0f}s budget)")
    entrypoint = f"python -c {shlex.quote(_SMOKE_SRC)}"
    result = await cluster.submit_and_tail(
        entrypoint,
        runtime_env={
            "env_vars": {
                "PREFLIGHT_WORLD": str(world),
                "PREFLIGHT_TIMEOUT_S": str(timeout_s),
                "PREFLIGHT_MASTER_ADDR": cluster.head_addr,
                **(env or {}),
            }
        },
    )
    if result.status != "SUCCEEDED":
        detail = result.message or f"status {result.status}"
        raise RuntimeError(
            f"Cluster preflight failed: {detail}. "
            "The gang could not complete a small all_gather/all_to_all; "
            "check the named ranks before retrying."
        )
