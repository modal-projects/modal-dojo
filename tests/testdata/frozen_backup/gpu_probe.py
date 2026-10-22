"""Real CUDA gate, invoked explicitly by modal/check_frozen_backup.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch_memory_saver import torch_memory_saver as saver


def main(mode, mutation):
    os.environ["TMS_DISK_BACKUP_CHUNK_MB"] = "1"
    saver.hook_mode = mode
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="tau-frozen-gpu-") as tmp:
        saver.set_disk_backup_dir(tmp)
        with saver.region("weights", enable_disk_backup=True):
            base = torch.randn((8192, 8192), device="cuda")
            tail = torch.arange(12345, device="cuda", dtype=torch.int64)
        with saver.region("adapters", enable_disk_backup=True):
            a = torch.randn((8192, 8), device="cuda") * 0.01
            b = torch.zeros((8, 8192), device="cuda")
        oracle = base.clone()
        tail_oracle = tail.cpu()
        x = torch.randn((4, 8192), device="cuda")

        def forward(weight):
            return (x @ weight + (x @ a) @ b).log_softmax(-1)  # noqa: F821 - deleted only during final pool cleanup

        sealed = saver.seal_disk_backup("weights")
        assert sealed >= base.nbytes + tail.nbytes
        if mutation.startswith("before"):
            print("ATTEMPTING_PROTECTED_WRITE", flush=True)
            if mutation.endswith("copy"):
                base.copy_(torch.zeros(base.shape))
            else:
                base.add_(1)
            torch.cuda.synchronize()
            raise AssertionError("protected base mutation succeeded")
        # Capture pointers once, then replay after every unmap/remap.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                forward(base)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = forward(base)
        initial = forward(base).clone()
        snapshots = None
        for step in range(3):
            b.add_(0.01)
            expected = forward(oracle).clone()
            assert not torch.equal(expected, initial), "adapter update had no effect"
            torch.cuda.synchronize()
            saver.seal_disk_backup("weights")
            saver.pause("weights")
            current = {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                for p in Path(tmp).iterdir()
                if snapshots is None or p.name in snapshots
            }
            if snapshots is None:
                snapshots = current
            assert current == snapshots, "frozen snapshots were rewritten"
            saver.pause("adapters")
            # Fill memory while the original addresses are unmapped.
            pressure = torch.full_like(oracle, 37)
            torch.cuda.synchronize()
            saver.resume("weights")
            saver.resume("adapters")
            del pressure
            assert torch.equal(base, oracle)
            assert torch.equal(tail.cpu(), tail_oracle)
            assert torch.equal(forward(base), expected), (
                "eager logprobs changed after restore"
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
            if mutation.startswith("after"):
                print("ATTEMPTING_PROTECTED_WRITE", flush=True)
                if mutation.endswith("copy"):
                    base.copy_(torch.zeros(base.shape))
                else:
                    base.add_(1)
                torch.cuda.synchronize()
                raise AssertionError("restored base mutation succeeded")
        # The upstream torch hook requires its region flag during allocator free.
        # Explicitly destroy graph references and cached pools before process exit.
        del graph, graph_output
        with saver.region("cleanup"):
            del base, tail, a, b
            saver._impl._mem_pools.clear()
        print(
            json.dumps(
                {
                    "mode": mode,
                    "cycles": 3,
                    "sealed_bytes": sealed,
                    "snapshot_files": len(snapshots),
                    "bitwise_equal": True,
                    "adapter_changes_preserved": True,
                    "graph_logprobs_equal": True,
                }
            )
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
