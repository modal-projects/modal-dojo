"""Patch Megatron's torch.py checkpoint loader for hybrid models.

1) Skip non-list values (BytesIO from _extra_state) in the rename loop.
   Without this, loading hybrid-model checkpoints crashes with
   "object of type '_io.BytesIO' has no len()".
2) Make _restore_dict_types tolerant of missing keys.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

p = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)
src = p.read_text()
patched = False

old1 = "        assert len(tensors) == len(rename_mapping[k])"
if old1 in src and "isinstance(tensors, list)" not in src:
    new1 = (
        "        if not isinstance(tensors, list):\n"
        "            continue  # skip BytesIO _extra_state entries\n"
        "        assert len(tensors) == len(rename_mapping[k])"
    )
    src = src.replace(old1, new1, 1)
    patched = True

m = re.search(r"^( +)_restore_dict_types\(x\[k\], v\)", src, re.MULTILINE)
if m and "k not in x" not in src:
    indent = m.group(1)
    old2 = indent + "_restore_dict_types(x[k], v)"
    new2 = (
        indent
        + "if k not in x:\n"
        + indent
        + "    if str(k) in x:\n"
        + indent
        + "        k = str(k)\n"
        + indent
        + "    else:\n"
        + indent
        + "        continue\n"
        + indent
        + "_restore_dict_types(x[k], v)"
    )
    src = src.replace(old2, new2, 1)
    patched = True

if patched:
    p.write_text(src)
    print("Patched torch.py for hybrid-model checkpoint loading")
