"""Small allocator correctness/performance proof before 64-GPU training."""

from pathlib import Path
import json
import modal

from configs.kimi_k3_b300 import retained_backup_image_commands

app = modal.App("k3-b300-memory-backup-proof")
image = (
    modal.Image.from_registry("radixark/miles:dev-202609251434")
    .entrypoint([])
    .run_commands(*retained_backup_image_commands())
)

PROGRAM = r"""
import gc, json, os, time
import torch
from torch_memory_saver import torch_memory_saver as saver
torch.cuda.set_device(0)
tag = "weights"
results = []
with saver.region(tag=tag, enable_cpu_backup=True):
    weight = torch.full((256 * 1024 * 1024,), 3.0, dtype=torch.float32, device="cuda")
with saver.region(tag="mutable", enable_cpu_backup=True):
    mutable = torch.full((1024 * 1024,), 5.0, device="cuda")
torch.cuda.synchronize()
for cycle in range(3):
    weight.fill_(3.0 + cycle)
    mutable.fill_(5.0 + cycle)
    torch.cuda.synchronize()
    started = time.monotonic()
    saver.pause(tag)
    pause_s = time.monotonic() - started
    backup = saver.get_cpu_backup(weight, zero_copy=True)
    assert backup[0].item() == 3.0 + cycle
    del backup
    saver.pause("mutable")
    started = time.monotonic()
    saver.resume(tag)
    resume_s = time.monotonic() - started
    saver.resume("mutable")
    assert torch.all(weight == 3.0 + cycle).item()
    assert torch.all(mutable == 5.0 + cycle).item()
    results.append(dict(cycle=cycle, pause_s=pause_s, resume_s=resume_s))
del weight, mutable
gc.collect()
torch.cuda.empty_cache()
print("RESULT " + json.dumps({"retained_tag": os.getenv("DOJO_TMS_RETAIN_BACKUP_TAG"), "cycles": results}))
"""


@app.function(
    image=image, gpu="B300", serialized=True, timeout=600, cpu=4, memory=16384
)
def check():
    import os
    import subprocess
    import sys
    from torch_memory_saver.utils import get_binary_path_from_package

    results = []
    for enabled in (False, True):
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(
            get_binary_path_from_package("torch_memory_saver_hook_mode_preload")
        )
        env["DOJO_TMS_RETAIN_BACKUP_TAG"] = "weights" if enabled else ""
        p = subprocess.run(
            [sys.executable, "-c", PROGRAM], env=env, text=True, capture_output=True
        )
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        results.append(p.stdout)
    return results


@app.local_entrypoint()
def main():
    results = check.remote()
    dest = Path(".modal-dojo/new_models/Kimi_K3_B300/memory-backup-proof.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2))
    for result in results:
        print(result)
