"""Backport NVIDIA/Megatron-LM #3845 into slime's pinned Megatron.

The async dist-checkpoint writer's ``_clone_if_needed`` (nested in
``FileSystemWriterAsync.write_preloaded_data`` in ``filesystem_async.py``)
returns GPU tensors untouched and defers the D2H copy to the async worker. For
quantized CUDA tensors (FP8/TE ``_extra_state``) the IPC handle isn't created,
so the async worker hits invalid memory access and the torch_dist save crashes
in ``inline_container.cc`` with "unexpected pos" (e.g. the GLM-5.2 convert).

Upstream #3845 dequantizes such tensors before they reach the writer. slime
pins a pre-#3845 Megatron, so we inject the same guard here: for a CUDA tensor
that exposes a ``dequantize`` method, dequantize before returning. No-op for
non-quantized tensors, so it's safe for every image.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

filesystem_async_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/filesystem_async.py"
)

if not filesystem_async_py.exists():
    print(f"WARNING: {filesystem_async_py} not found — skipping dist-ckpt patch")
    raise SystemExit(0)

src = filesystem_async_py.read_text()
marker = "PATCHED_DEQUANTIZE_QUANTIZED"

# Insert the dequantize guard as the first statement inside the
# ``if ten.device.type != "cpu":`` branch of the clone helper. Detect the
# branch's indentation from the source so the injected lines line up.
m = re.search(r'^( +)if ten\.device\.type != "cpu":\n', src, re.MULTILINE)
if m and marker not in src:
    if_indent = m.group(1)
    body_indent = if_indent + "    "
    injected = (
        f"{body_indent}# {marker}: dequantize quantized CUDA tensors (FP8/TE) so the\n"
        f"{body_indent}# async writer doesn't hit an invalid IPC handle. Backport of\n"
        f"{body_indent}# NVIDIA/Megatron-LM#3845; no-op for non-quantized tensors.\n"
        f'{body_indent}if ten.device.type == "cuda" and "dequantize" in type(ten).__dict__:\n'
        f"{body_indent}    ten = ten.dequantize()\n"
    )
    src = src.replace(m.group(0), m.group(0) + injected, 1)
    filesystem_async_py.write_text(src)
    print("Patched filesystem_async.py to dequantize quantized CUDA tensors")
elif marker in src:
    print("filesystem_async.py already patched for quantized CUDA tensors")
else:
    print(
        "WARNING: could not patch filesystem_async.py — "
        'target `if ten.device.type != "cpu":` not found'
    )
