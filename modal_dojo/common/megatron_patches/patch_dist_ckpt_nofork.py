"""Stop Megatron's torch_dist writer from ``os.fork()``ing out of a Ray actor.

Slime pins a pre-#3633 Megatron whose ``write_preloaded_data_multiproc``
does ``mp.get_context("fork"); ctx.Process(...); p.start()``. From a
multithreaded CUDA/Ray actor that ``os.fork()`` raises
``BlockingIOError: [Errno 11] Resource temporarily unavailable`` (EAGAIN)
on the final checkpoint save. ``--dist-ckpt-workers`` landed in the same
PR that deleted this function, so the gym cannot pass the flag.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

filesystem_async_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)

if not filesystem_async_py.exists():
    print(f"WARNING: {filesystem_async_py} not found — skipping dist-ckpt nofork patch")
    raise SystemExit(0)

src = filesystem_async_py.read_text()
marker = "PATCHED_DIST_CKPT_NOFORK"

if marker in src:
    print("filesystem_async.py already patched to write torch_dist in-process")
    raise SystemExit(0)

if "write_preloaded_data_multiproc" not in src:
    print(
        "filesystem_async.py has no write_preloaded_data_multiproc — "
        "skipping dist-ckpt nofork patch"
    )
    raise SystemExit(0)

spawn = re.search(
    r"^([ \t]+)p_list\.append\(\s*"
    r"ctx\.Process\(\s*"
    r"target=partial\(FileSystemWriterAsync\.write_preloaded_data,"
    r"\s*transform_list\),\s*"
    r"kwargs=kwargs,\s*"
    r"\)\s*\)$",
    src,
    re.MULTILINE,
)
start = re.search(r"^([ \t]+)for p in p_list:\n\1    p\.start\(\)$", src, re.MULTILINE)
join = re.search(r"^([ \t]+)p_list\[local_proc_idx\]\.join\(\)$", src, re.MULTILINE)

if not (spawn and start and join):
    print(
        "WARNING: could not patch filesystem_async.py — "
        "torch_dist Process spawn / p.start() / join() not found"
    )
    raise SystemExit(0)

indent = spawn.group(1)
src = src.replace(
    spawn.group(0),
    (
        f"{indent}FileSystemWriterAsync.write_preloaded_data(\n"
        f"{indent}    transform_list, **kwargs\n"
        f"{indent})  # {marker}"
    ),
    1,
)
src = src.replace(start.group(0), f"{start.group(1)}pass  # {marker}", 1)
src = src.replace(join.group(0), f"{join.group(1)}pass  # {marker}", 1)
filesystem_async_py.write_text(src)
print("Patched filesystem_async.py to write torch_dist in-process (no fork)")
