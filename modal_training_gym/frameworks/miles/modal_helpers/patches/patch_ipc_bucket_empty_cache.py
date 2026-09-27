"""Return the trainer's freed sync buckets to CUDA as each one is sent.

``miles/backends/training_utils/weight_update/protocols/cuda_ipc.py`` streams
the adapter to the colocated engines one flattened bucket at a time and drops
each bucket once the engine has cloned it, but only calls
``torch.cuda.empty_cache()`` after the engines resume. The buckets and the
gathered HF tensors behind them are all different sizes, so the caching
allocator cannot reuse the freed blocks and keeps reaching for new segments:
on Kimi-K3 the trainer's reserved memory grew by ~52 GB over 276 buckets while
its live allocations stayed flat. At the first sync the engine is fully
resident and stashes the whole ~54 GB adapter on top of its weights, so that
cache growth is exactly what pushes the 267 GB B300 over the edge. Emptying
the cache after every bucket keeps the trainer's footprint at one bucket;
blocks the engine still maps over IPC are refcounted and stay put.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

MARKER = "PATCHED_IPC_BUCKET_EMPTY_CACHE"

TARGET = pathlib.Path(
    "/root/miles/miles/backends/training_utils/weight_update/protocols/cuda_ipc.py"
)
DEL_RE = re.compile(r"^( +)del long_lived_tensors$", re.MULTILINE)

if not TARGET.exists():
    print(f"{TARGET} not found; skipping IPC bucket empty_cache patch")
    raise SystemExit(0)

src = TARGET.read_text()
if MARKER in src:
    print("IPC bucket empty_cache patch already applied")
    raise SystemExit(0)

src, replacements = DEL_RE.subn(
    rf"\1del long_lived_tensors\n\1torch.cuda.empty_cache()  # {MARKER}", src
)
if replacements != 1:
    raise SystemExit(
        "IPC bucket empty_cache patch did not match exactly once; miles' "
        "cuda_ipc.py has changed. Re-check send_bucket before shipping."
    )

TARGET.write_text(src)
print("Patched cuda_ipc.send_bucket to empty the CUDA cache after each bucket")
