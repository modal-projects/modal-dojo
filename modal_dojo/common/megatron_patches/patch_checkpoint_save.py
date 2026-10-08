"""Patch Megatron/PyTorch checkpoint save for hybrid MoE models.

The ``inline_container.cc`` "unexpected pos" error occurs when PyTorch's
zip writer tracks a file position that diverges from what miniz expects.
Root cause: ``_mcore_to_torch_sharded_object`` in Megatron's ``torch.py``
returns ``BytesIO`` objects with the cursor at the end.  Downstream code
(plan size calculation, ``getbuffer()``) *usually* tolerates this, but
certain PyTorch versions and write-path combinations do not.

This patch applies two fixes:
1. Seek every ``BytesIO`` returned by ``_mcore_to_torch_sharded_object``
   back to position 0 so all consumers see consistent data.
2. Wrap ``MCoreSavePlanner.transform_object`` to seek(0) any ``BytesIO``
   before the planner hands it to the writer, as a defence-in-depth
   measure.

Executed at image-build time via ``python3 <this file>``.
"""

import pathlib
import re

# ── 1. Patch torch.py: seek(0) in _mcore_to_torch_sharded_object ────────────

torch_py = pathlib.Path(
    "/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py"
)
src = torch_py.read_text()
patched = False

# The function creates a BytesIO via torch.save() and returns it with
# the cursor at end-of-stream.  Add a seek(0) before the return.
# Use regex to detect the actual indentation of the return statement.
marker = "PATCHED_SEEK_SHOBJ"
m = re.search(r"^( +)(return serialized_data)\b", src, re.MULTILINE)
if m and marker not in src:
    indent = m.group(1)
    old_line = m.group(0)
    new_line = (
        f"{indent}serialized_data.seek(0)  # {marker}\n{indent}return serialized_data"
    )
    src = src.replace(old_line, new_line, 1)
    patched = True

# ── 2. Patch MCoreSavePlanner.transform_object to seek(0) BytesIO ───────────
# The slime fork's transform_object is a simple one-liner:
#     def transform_object(self, write_item: WriteItem, object: Any):
#         return object

marker2 = "PATCHED_TRANSFORM_SEEK"
if marker2 not in src:
    # Match the transform_object method and detect its indentation
    m2 = re.search(
        r"^( +)def transform_object\(self, write_item: WriteItem, object: Any\):\n"
        r"\1    return object",
        src,
        re.MULTILINE,
    )
    if m2:
        indent = m2.group(1)
        old_block = m2.group(0)
        new_block = (
            f"{indent}def transform_object(self, write_item: WriteItem, object: Any):\n"
            f"{indent}    import io as _io  # {marker2}\n"
            f"{indent}    if isinstance(object, _io.BytesIO):\n"
            f"{indent}        object.seek(0)\n"
            f"{indent}    return object"
        )
        src = src.replace(old_block, new_block, 1)
        patched = True

if patched:
    torch_py.write_text(src)
    print("Patched torch.py for checkpoint save BytesIO handling")
else:
    print("WARNING: Could not patch torch.py — target strings not found")
